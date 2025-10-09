#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast(er) ES fine-tuning loop with LoRA + Accelerate
- Single model per process (no thread-per-seed replicas)
- Mini-batch evaluation per iteration (batched generation)
- Optional FlashAttention-2
- Minimal synchronization/logging in the hot path
"""

import argparse
import gc
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# ---------- Args ----------
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='hf_cache')
parser.add_argument('--precision', type=str, choices=['bf16', 'fp16', 'fp32'], default='bf16')
parser.add_argument('--data_path', type=str, default='data/countdown.json')
parser.add_argument('--data_sample', type=int, default=1000, help='Max dataset items to load')
parser.add_argument('--mini_batch', type=int, default=32, help='Prompts evaluated per iteration (B)')
parser.add_argument('--num_iterations', type=int, default=1000)
parser.add_argument('--population_size', type=int, default=30)
parser.add_argument('--sigma', type=float, default=1e-3)
parser.add_argument('--alpha', type=float, default=5e-4)
parser.add_argument('--max_new_tokens', type=int, default=128)
parser.add_argument('--do_sample', action='store_true', help='Use sampling instead of greedy')
parser.add_argument('--seed', type=int, default=33)

# LoRA
parser.add_argument('--lora_r', type=int, default=16)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--lora_targets', nargs='*',
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

# Perf toggles / logging
parser.add_argument('--use_flash_attn', action='store_true', help='Try to enable FlashAttention-2')
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--wandb_project', type=str, default='es-fine-tuning-fast')
parser.add_argument('--wandb_run_name', type=str, default=None)
parser.add_argument('--track_flops', action='store_true', help='Rough FLOPs estimation (adds small overhead)')
parser.add_argument('--log_every', type=int, default=1)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# ---------- Small helpers ----------
torch.backends.cuda.matmul.allow_tf32 = True

def dprint(*a, **k):
    if args.verbose:
        print(*a, **k)

def force_memory_cleanup(light=True):
    gc.collect()
    if torch.cuda.is_available():
        if not light:
            torch.cuda.empty_cache()

def estimate_generation_flops(model, input_len, output_len, batch_size=1):
    # Crude but useful: ~2 * params * (prefill + decode context) * batch
    nparams = sum(p.numel() for p in model.parameters())
    prefill = 2 * nparams * input_len * batch_size
    # Decode attends to growing context
    decode = 0
    for i in range(output_len):
        decode += 2 * nparams * (input_len + i) * batch_size
    return prefill + decode

# Reward function for the countdown task (same as your original import)
def reward_function(model_response: str, numbers, target):
    # Minimal placeholder — replace with your countdown_task.reward_function if needed.
    # Expect dict: {"reward": float}
    # For speed demo, give partial credit if target integer appears.
    try:
        contains = str(target) in model_response if target is not None else False
        return {"reward": 1.0 if contains else 0.0}
    except Exception:
        return {"reward": 0.0}

# ---------- Data ----------
def load_countdown_dataset(path, limit):
    with open(path, 'r') as f:
        data_json = json.load(f)
    ds = []
    for item in data_json:
        ds.append((item['context'], item['target']))
        if len(ds) >= limit:
            break
    return ds

# ---------- Model ----------
def load_tokenizer(model_name, cache_dir):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok

def load_lora_model(model_name, cache_dir, precision, lora_cfg, use_flash_attn=False, device_map=None):
    dtype = torch.float32
    if precision == 'bf16':
        dtype = torch.bfloat16
    elif precision == 'fp16':
        dtype = torch.float16

    kwargs = dict(cache_dir=cache_dir, torch_dtype=dtype)
    if device_map is not None:
        kwargs["device_map"] = device_map

    if use_flash_attn:
        # Try FA2; if not supported, fall back silently
        try:
            kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass

    base = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = get_peft_model(base, lora_cfg)
    model.eval()
    return model

# ---------- ES Core ----------
@torch.inference_mode()
def batched_generate_rewards(model, tokenizer, inputs_text, targets, max_new_tokens, do_sample, track_flops=False):
    # Tokenize as a batch
    batch = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True)
    batch = {k: v.to(model.device) for k, v in batch.items()}

    input_len = batch["input_ids"].shape[1]
    outputs = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Decode
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Compute rewards
    rewards = []
    for resp, tgt, inp in zip(texts, targets, inputs_text):
        numbers = None
        if "[" in inp and "]" in inp:
            try:
                s = inp[inp.find('[')+1:inp.find(']')]
                numbers = [int(n) for n in s.split() if n.strip().isdigit()]
            except Exception:
                numbers = None
        target = int(tgt) if isinstance(tgt, (int, str)) and str(tgt).isdigit() else None
        r = reward_function(resp, numbers, target)["reward"]
        rewards.append(float(r))

    flops = 0
    if track_flops:
        out_len = outputs.shape[1] - input_len
        flops = estimate_generation_flops(model, input_len, out_len, batch_size=len(inputs_text))
    return np.array(rewards, dtype=np.float32), flops

def apply_noise_lora(model, sigma, seed):
    # Only LoRA trainable params (requires_grad True)
    g = None
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if g is None:
            g = torch.Generator(device=p.device)
            g.manual_seed(int(seed))
        noise = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)
        p.data.add_(sigma * noise)

def remove_noise_lora(model, sigma, seed):
    # Regenerate and subtract
    g = None
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if g is None:
            g = torch.Generator(device=p.device)
            g.manual_seed(int(seed))
        noise = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)
        p.data.add_(-sigma * noise)

def es_update_lora(model, sigma, alpha, seeds, rewards_norm):
    # Regenerate noise per param and accumulate update
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = torch.Generator(device=p.device)
        upd = torch.zeros_like(p)
        for seed, r in zip(seeds, rewards_norm):
            g.manual_seed(int(seed))
            noise = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)
            upd.add_(noise.mul(float(r)))
        upd.div_(len(seeds))
        p.data.add_(alpha * upd)
        # let GC reclaim
        del upd
    force_memory_cleanup(light=True)

# ---------- Main ----------
def main():
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    # Seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    if is_main:
        print(f"Loading dataset: {args.data_path} (limit {args.data_sample})")
    dataset = load_countdown_dataset(args.data_path, args.data_sample)
    if is_main:
        print(f"Loaded {len(dataset)} samples")

    # Tokenizer & model
    tok = load_tokenizer(args.model_name, args.hf_cache_dir)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets,
        bias="none",
        inference_mode=False,
    )

    # Map this process to its local device
    device_map = {"": accelerator.process_index}
    model = load_lora_model(
        args.model_name,
        cache_dir=args.hf_cache_dir,
        precision=args.precision,
        lora_cfg=lora_cfg,
        use_flash_attn=args.use_flash_attn,
        device_map=device_map
    )

    # WANDB
    if args.use_wandb and WANDB_AVAILABLE and is_main:
        run_name = args.wandb_run_name or f"es_fast_{args.model_name.replace('/', '_')}_pop{args.population_size}_B{args.mini_batch}_mx{args.max_new_tokens}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=dict(
                model_name=args.model_name,
                population_size=args.population_size,
                num_iterations=args.num_iterations,
                sigma=args.sigma,
                alpha=args.alpha,
                max_new_tokens=args.max_new_tokens,
                mini_batch=args.mini_batch,
                precision=args.precision,
                use_flash_attn=args.use_flash_attn,
            ),
        )

    accelerator.wait_for_everyone()
    total_flops = 0.0
    t0 = time.time()

    for it in range(1, args.num_iterations + 1):
        iter_t0 = time.time()

        # Sample a small random mini-batch each iter
        mb = random.sample(dataset, k=min(args.mini_batch, len(dataset)))
        inputs = [x[0] for x in mb]
        targets = [x[1] for x in mb]

        # Generate seeds on main; broadcast
        if is_main:
            seeds = np.random.randint(0, 2**30, size=args.population_size, dtype=np.int64).tolist()
            seeds_t = torch.tensor(seeds, device=accelerator.device)
        else:
            seeds_t = torch.zeros(args.population_size, dtype=torch.long, device=accelerator.device)

        if accelerator.num_processes > 1:
            torch.distributed.broadcast(seeds_t, src=0)
        seeds = seeds_t.cpu().tolist()
        del seeds_t

        # Each process evaluates a disjoint subset of seeds (round-robin)
        local_indices = [i for i in range(args.population_size) if i % accelerator.num_processes == accelerator.process_index]

        local_rewards = np.zeros(args.population_size, dtype=np.float32)
        local_flops = 0.0

        for idx in local_indices:
            seed = seeds[idx]
            # Apply noise
            apply_noise_lora(model, args.sigma, seed)
            # Batched generation & rewards
            rewards, flops = batched_generate_rewards(
                model, tok, inputs, targets,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                track_flops=args.track_flops
            )
            # Remove noise
            remove_noise_lora(model, args.sigma, seed)
            # Aggregate
            local_rewards[idx] = float(rewards.mean())
            if args.track_flops:
                local_flops += float(flops)

        # Reduce (sum) across processes
        rewards_all = torch.tensor(local_rewards, device=accelerator.device)
        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(rewards_all, op=torch.distributed.ReduceOp.SUM)

        rewards = rewards_all.cpu().numpy()
        del rewards_all

        # Normalize rewards
        mean_r = float(rewards.mean())
        std_r = float(rewards.std() + 1e-8)
        rewards_norm = (rewards - mean_r) / std_r

        # ES update on LoRA params (regenerate noises)
        es_update_lora(model, args.sigma, args.alpha, seeds, rewards_norm)

        iter_time = time.time() - iter_t0
        if args.track_flops:
            # Sum FLOPs across processes
            flops_t = torch.tensor(local_flops, device=accelerator.device, dtype=torch.float64)
            if accelerator.num_processes > 1:
                torch.distributed.all_reduce(flops_t, op=torch.distributed.ReduceOp.SUM)
            iter_flops = float(flops_t.item())
            total_flops += iter_flops
        else:
            iter_flops = 0.0

        if is_main and (it % args.log_every == 0):
            log_line = f"[Iter {it}/{args.num_iterations}] time={iter_time:.2f}s, mean={mean_r:.4f}, std={std_r:.4f}, max={rewards.max():.4f}, min={rewards.min():.4f}"
            if args.track_flops:
                log_line += f", TFLOPs={iter_flops/1e12:.2f}"
            print(log_line)

            if args.use_wandb and WANDB_AVAILABLE:
                log_dict = {
                    "iteration": it,
                    "reward/mean": mean_r,
                    "reward/std": std_r,
                    "reward/max": float(rewards.max()),
                    "reward/min": float(rewards.min()),
                    "time/iter_seconds": iter_time,
                }
                if args.track_flops:
                    log_dict["compute/iter_tflops"] = iter_flops / 1e12
                    log_dict["compute/total_tflops"] = total_flops / 1e12
                    log_dict["compute/tflops_per_second"] = (iter_flops / 1e12) / max(iter_time, 1e-6)
                if torch.cuda.is_available():
                    log_dict["memory/gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
                    log_dict["memory/gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                wandb.log(log_dict)

        # light GC
        force_memory_cleanup(light=True)

    total_time = time.time() - t0
    if is_main:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} min)")
        if args.track_flops:
            print(f"Total TFLOPs: {total_flops/1e12:.2f}")
        # Save adapters only (fast + small)
        save_dir = f"{args.model_name}_es_fast_pop{args.population_size}_B{args.mini_batch}_mx{args.max_new_tokens}_{args.precision}_final"
        print(f"Saving adapters to: {save_dir}")
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()
