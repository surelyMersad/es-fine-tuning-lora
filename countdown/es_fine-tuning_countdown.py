import argparse
import copy
import gc
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from accelerate import Accelerator
# === LoRA / PEFT ===
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
# --- vLLM integration BEGIN ---
try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None
# --- vLLM integration END ---


logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='hf_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--gpu_threads', type=int, default=4, help='Number of parallel threads per GPU')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--data_sample', type=int, default=1000, help='Number of data samples to use for training')
parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='es-fine-tuning-countdown', help='Wandb project name')
parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name (default: auto-generated)')
parser.add_argument('--track_flops', action='store_true', help='Track FLOPs for forward passes')
parser.add_argument('--use_lora', action='store_true', default=False, help='Use LoRA for fine-tuning (default: True)')
# --- vLLM integration BEGIN ---
parser.add_argument('--use_vllm_rollouts', action='store_true',
                    help='Use vLLM for batched generation rollouts (HF model remains the one being updated).')
# --- vLLM integration END ---

args = parser.parse_args()

# Hyperparameters for ES
NUM_ITERATIONS = 1000             # Number of ES iterations (generations)
POPULATION_SIZE = 30              # Population size (number of perturbations per iteration)
SIGMA = 0.001                     # Standard deviation for weight perturbations (noise scale)
ALPHA = 0.0005                    # Learning rate
max_new_tokens = 1024             # Maximum number of tokens allowed to be generated
do_sample = False                 # Greedy decoding for ES
initial_seed = 33                 # Initial random seed

# === LoRA hyperparams ===
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Qwen/LLaMA-style module names
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- vLLM integration BEGIN ---
USE_VLLM = args.use_vllm_rollouts
VLLM_LLM = None
VLLM_MODEL = None
VLLM_SAMPLING = None
VLLM_LOCK = threading.Lock()  # serialize weight loads + generate calls

def _init_vllm(model_name: str, accelerator):
    """
    Initialize a single vLLM engine per process, matching Accelerate ranks.
    """
    if LLM is None:
        raise RuntimeError("vLLM is not installed but --use_vllm_rollouts was set.")

    # Align vLLM's distributed executor env with Accelerate ranks
    os.environ.setdefault('RANK', str(accelerator.process_index))
    os.environ.setdefault('LOCAL_RANK', str(getattr(accelerator, "local_process_index", accelerator.process_index)))
    os.environ.setdefault('WORLD_SIZE', str(accelerator.num_processes))
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.80,
        distributed_executor_backend="external_launcher",
        enable_sleep_mode=True,
    )
    sampling = SamplingParams(
        max_tokens=max_new_tokens,           # reuse your global
        temperature=0.0 if not do_sample else 1.0,
        top_p=1.0,
        n=1
    )
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    return llm, model, sampling

@torch.no_grad()
def _collect_trainable_named_tensors(hf_model):
    if args.use_lora:
        hf_model.merge_adapter()
        print(f"[DEBUG] LoRA model merged")
    items = []
    for n, p in hf_model.named_parameters():
        if not p.requires_grad:
            # print(f"[DEBUG] Collecting non-trainable param {n}")
            if args.use_lora:
                to_cut = len("base_model.model.")
                n = n[to_cut:]
            items.append((n, p.detach()))
    if args.use_lora:
        hf_model.unmerge_adapter()
        print(f"[DEBUG] LoRA model unmerged")
    print(f"[DEBUG] Collected {len(items)} trainable tensors from HF model")
    # print(f"[DEBUG] Items: {items}")
    return items

def _sync_hf_to_vllm(hf_model):
    print(f"[DEBUG] Starting HF to vLLM weight sync...")
    # Build map of vLLM param names
    vllm_names = {n for n, _ in VLLM_MODEL.named_parameters()} if hasattr(VLLM_MODEL, "named_parameters") else None
    raw = _collect_trainable_named_tensors(hf_model)
    print(f"[DEBUG] Collected {len(raw)} trainable tensors from HF model")
    
    if vllm_names is not None:
        print(f"[DEBUG] vLLM has {len(vllm_names)} parameters")
        # Print first few parameter names for debugging
        
        
        to_load = [(n, t) for (n, t) in raw if n in vllm_names]
    else:
        print(f"[DEBUG] vLLM model has no named_parameters, loading all")
        to_load = raw
        
    print(f"[DEBUG] Loading {len(to_load)} tensors to vLLM")

    # Some builds want list-of-lists; this keeps compatibility minimal
    try:
        VLLM_MODEL.load_weights(to_load)
        print(f"[DEBUG] vLLM weight loading successful (list format)")
    except Exception as e:
        print(f"[DEBUG] vLLM weight loading failed with list format, trying list-of-lists: {e}")
        VLLM_MODEL.load_weights([[n, t] for (n, t) in to_load])
        print(f"[DEBUG] vLLM weight loading successful (list-of-lists format)")
# --- vLLM integration END ---


# Import countdown reward function
from countdown_task import reward_function

print("Using countdown reward function")

# Dataset will be loaded in main function

def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def estimate_generation_flops(num_params, input_length, output_length, batch_size=1):
    """
    Estimate FLOPs for autoregressive generation.
    
    For transformer models:
    - Each forward pass through a token position ≈ 2 * num_params FLOPs
    - For generation: we do (input_length + output_length) forward passes
    - Each subsequent token requires attending to all previous tokens
    
    Uses closed-form formula for efficiency.
    """
    # Prefill: input_length tokens in parallel
    prefill_flops = 2 * num_params * input_length * batch_size
    
    # Decode: output_length tokens sequentially, each attending to growing context
    # Sum of (input_length + i) for i in range(output_length)
    # = output_length * input_length + sum(0 to output_length-1)
    # = output_length * input_length + output_length * (output_length - 1) / 2
    decode_flops = 2 * num_params * batch_size * (
        output_length * input_length + output_length * (output_length - 1) // 2
    )
    
    total_flops = prefill_flops + decode_flops
    return total_flops

def save_model_checkpoint(model, tokenizer, iteration, model_name, initial_seed, args, dataset_size):
    """Save LoRA adapter checkpoint at specified iteration"""
    question_num = dataset_size
    save_dir = f"{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{iteration}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_question_num{question_num}_checkpoint"
    print(f"Saving checkpoint at iteration {iteration} to {save_dir}...")
    # For PeftModel this saves adapters only (desired)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Checkpoint saved successfully.")

def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False, track_flops=False, num_params=None):
    """
    Generate a response from the model given an input (single or batch) and compute rewards.
    """
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    # Handle both single input and batch input
    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    if USE_VLLM:
        # vLLM takes raw prompts; we DO NOT retokenize for generation.
        # (We still use tokenizer later for FLOPs estimation if requested.)
        with torch.inference_mode():
            # # Ensure engine is awake
            # try:
            #     VLLM_LLM.wake_up()
            # except Exception:
            #     pass

            outputs = VLLM_LLM.generate(
                prompts=input_texts,
                sampling_params=VLLM_SAMPLING,
                use_tqdm=False
            )
            # Each element has .outputs[0].text
            generated_texts = [o.outputs[0].text if o.outputs else "" for o in outputs]

            # # Optional: nap to free scheduler work
            # try:
            #     VLLM_LLM.sleep(level=1)
            # except Exception:
            #     pass

        # FLOPs estimate (optional) using tokenizer lengths
        flops = 0
        if track_flops and num_params is not None:
            # Tokenize once to get lengths
            input_len = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")["input_ids"].shape[1]
            # Rough: tokenize outputs to get length
            out_tok = tokenizer(generated_texts, padding=True, truncation=True, return_tensors="pt")["input_ids"]
            output_length = out_tok.shape[1]
            batch_size = len(input_texts)
            flops = estimate_generation_flops(num_params, input_len, output_length, batch_size)
    else:
        # Original HF path
        tokenized_inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
        attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)

        input_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(accelerator.device)

        flops = 0
        if track_flops and num_params is not None:
            output_length = outputs.shape[1] - input_length
            flops = estimate_generation_flops(num_params, input_length, output_length, batch_size)

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        del input_ids, outputs, attention_mask, tokenized_inputs
        torch.cuda.empty_cache()

    

    # Compute rewards for batch texts
    rewards = []
    for i, (gen_text, tgt_text, inp_text) in enumerate(zip(generated_texts, target_texts, input_texts)):
        if verbose and i == 0:  # Log first example for debugging
            print(f"[DEBUG] Process {accelerator.process_index} Thread {thread_id}: Computing reward for example {i}")
            print(f"[DEBUG] Input: {inp_text[:100]}...")
            print(f"[DEBUG] Generated: {gen_text[:100]}...")
            print(f"[DEBUG] Target: {tgt_text}")
        
        numbers = None
        target = None
        if "[" in inp_text and "]" in inp_text:
            start_idx = inp_text.find("[")
            end_idx = inp_text.find("]")
            if start_idx != -1 and end_idx != -1:
                numbers_str = inp_text[start_idx+1:end_idx]
                numbers = [int(n) for n in numbers_str.split() if n.isdigit()]

        if isinstance(tgt_text, str) and tgt_text.isdigit():
            target = int(tgt_text)
        elif isinstance(tgt_text, int):
            target = tgt_text

        model_response = gen_text
        if "assistant:" in gen_text:
            model_response = gen_text.split("assistant:")[-1].strip()

        # Use reward_function from countdown_task.py
        reward_result = reward_function(model_response, numbers, target)
        reward = reward_result["reward"]
        rewards.append(reward)
        
        if verbose and i == 0:  # Log first reward for debugging
            print(f"[DEBUG] Reward: {reward}, Numbers: {numbers}, Target: {target}")

    if return_text:
        if track_flops:
            return rewards, generated_texts, flops
        else:
            return rewards, generated_texts
    else:
        if track_flops:
            return rewards, flops
        else:
            return rewards

def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose, dataset, track_flops, num_params = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Apply noise to trainable params (LoRA params if using LoRA, or all params if not)
    num_params_perturbed = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        num_params_perturbed += 1
        
        # Store original value for debugging
        original_sum = param.data.sum().item() if verbose and num_params_perturbed == 1 else None
        
        if verbose and num_params_perturbed <= 3:  # Log first few params for debugging
            print(f"[DEBUG] Process {accelerator.process_index} Thread {thread_id}: Perturbing param {name}, shape: {param.shape}, device: {param.device}")
            print(f"[DEBUG] Original param sum: {original_sum}")
        
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(SIGMA * noise)
        
        # Log change for first param
        if verbose and num_params_perturbed == 1:
            new_sum = param.data.sum().item()
            noise_sum = noise.sum().item()
            print(f"[DEBUG] After perturbation - param sum: {new_sum}, noise sum: {noise_sum}, change: {new_sum - original_sum}")
    
    if verbose:
        print(f"[DEBUG] Process {accelerator.process_index} Thread {thread_id}: Perturbed {num_params_perturbed} trainable parameters")

    # Ensure weights are fully loaded before evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # Evaluate all prompts with perturbed weights in batch
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]

    if USE_VLLM:
        # Push *perturbed* weights to vLLM and rollout under a lock
        if verbose:
            print(f"[DEBUG] Process {accelerator.process_index} Thread {thread_id}: Syncing perturbed weights to vLLM")
        with VLLM_LOCK:
            _sync_hf_to_vllm(model)

    
    eval_result = evaluate_model(
        model, tokenizer, input_texts, target_texts, accelerator,
        seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, return_text=False, track_flops=track_flops, num_params=num_params
    )

    
    if track_flops:
        rewards, flops = eval_result
    else:
        rewards = eval_result
        flops = 0
    
    total_reward = sum(rewards)

    # Restore original weights (subtract the same noise) for trainable params
    num_params_restored = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        num_params_restored += 1
        if verbose and num_params_restored <= 3:  # Log first few params for debugging
            print(f"[DEBUG] Process {accelerator.process_index} Thread {thread_id}: Restoring param {name}")
        
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(-SIGMA * noise)
    
    if verbose:
        print(f"[DEBUG] Process {accelerator.process_index} Thread {thread_id}: Restored {num_params_restored} trainable parameters")
    
   
    

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    if USE_VLLM:
        # After restoring HF to baseline, sync baseline back to vLLM
        if verbose:
            print(f"[DEBUG] Process {accelerator.process_index} Thread {thread_id}: Syncing restored baseline weights to vLLM")
        with VLLM_LOCK:
            _sync_hf_to_vllm(model)
    

    average_reward = total_reward / len(dataset)

    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")

    if track_flops:
        return seed_idx, average_reward, flops
    else:
        return seed_idx, average_reward

# --- Main Evolution Strategies Loop ---
def main():
    accelerator = Accelerator()

    # --- Load Dataset from JSON File ---
    data_path = os.path.join(os.path.dirname(__file__), 'data/countdown.json')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data_json = json.load(f)

    dataset = []
    for item in data_json:
        context = item['context']
        target = item['target']
        dataset.append((context, target))

    # Use a subset of the dataset for training
    dataset = dataset[:args.data_sample]
    if accelerator.is_main_process:
        print(f"Loaded {len(dataset)} countdown samples from {data_path}")

    # Define model parameters early for wandb
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    if accelerator.is_main_process:
        print(f"Total processes: {accelerator.num_processes}, GPU threads per process: {args.gpu_threads}")
        print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
        print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")
        
        # Initialize wandb
        if args.use_wandb:
            lora_suffix = "_lora" if args.use_lora else "_full"
            wandb_run_name = args.wandb_run_name or f"es_{model_name.replace('/', '_')}_pop{POPULATION_SIZE}_sigma{SIGMA}_alpha{ALPHA}_seed{initial_seed}{lora_suffix}"
            config_dict = {
                "model_name": model_name,
                "population_size": POPULATION_SIZE,
                "num_iterations": NUM_ITERATIONS,
                "sigma": SIGMA,
                "alpha": ALPHA,
                "max_new_tokens": max_new_tokens,
                "initial_seed": initial_seed,
                "use_lora": args.use_lora,
                "precision": args.precision,
                "gpu_threads": args.gpu_threads,
                "num_gpus": accelerator.num_processes,
                "data_samples": len(dataset),
            }
            if args.use_lora:
                config_dict.update({
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "lora_dropout": LORA_DROPOUT,
                })
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                config=config_dict
            )
            print(f"Wandb initialized: {wandb.run.url}")

    # Load model

    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")

    # Load tokenizer once and set padding
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # === Build LoRA config once (if using LoRA) ===
    lora_cfg = None
    if args.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            inference_mode=False,
        )

    # Load base model per thread, optionally wrap with LoRA, eval mode for deterministic ES
    model_list = []
    for model_index in range(args.gpu_threads):
        print(f"[DEBUG] Process {accelerator.process_index}: Loading model {model_index+1}/{args.gpu_threads}")
        
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},  # Assign devices explicitly
            torch_dtype=torch.float16 if args.precision == 'fp16' else (torch.bfloat16 if args.precision == 'bf16' else torch.float32),
        )
        print(f"[DEBUG] Process {accelerator.process_index}: Base model loaded, device: {next(base.parameters()).device}")
        
        if args.use_lora:
            model = get_peft_model(base, lora_cfg)  # wrap with LoRA
            print(f"[DEBUG] Process {accelerator.process_index}: LoRA model wrapped")
        else:
            model = base  # use full model
            print(f"[DEBUG] Process {accelerator.process_index}: Using full model (no LoRA)")
            
        model.eval()
        print(f"[DEBUG] Process {accelerator.process_index}: Model set to eval mode")
        
        # Print trainable params for debugging
        if accelerator.is_main_process and model_index == 0:
            if args.use_lora:
                model.print_trainable_parameters()
                print(f"[DEBUG] LoRA trainable parameters printed above")
            else:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"[DEBUG] Full model - Total: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
        
        model_list.append(model)
        print(f"[DEBUG] Process {accelerator.process_index}: Model {model_index+1} added to model_list")
    if USE_VLLM:
        if accelerator.is_main_process:
            print("[DEBUG] Initializing vLLM engine for rollouts...")
        global VLLM_LLM, VLLM_MODEL, VLLM_SAMPLING
        VLLM_LLM, VLLM_MODEL, VLLM_SAMPLING = _init_vllm(model_name, accelerator)
        print(f"[DEBUG] Process {accelerator.process_index}: vLLM engine initialized")

        # Optional: prime vLLM with current baseline HF weights (trainable-only)
        print(f"[DEBUG] Process {accelerator.process_index}: Syncing HF weights to vLLM...")
        with VLLM_LOCK:
            _sync_hf_to_vllm(model_list[0])
        if accelerator.is_main_process:
            print("[DEBUG] vLLM ready (external launcher) and synced with HF baseline.")

    if accelerator.is_main_process:
        print("Model loaded successfully")
        if args.use_lora:
            print(f"Using LoRA with r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
        else:
            print("Using full model (no LoRA)")

    force_memory_cleanup()

    # Record total training start time
    training_start_time = time.time()
    
    # Initialize FLOPs tracking
    total_flops = 0
    # Cache parameter count for FLOPs calculation (compute once, reuse)
    num_params = sum(p.numel() for p in model_list[0].parameters()) if args.track_flops else None
    if args.track_flops and accelerator.is_main_process:
        print(f"Model has {num_params/1e9:.2f}B parameters")

    np.random.seed(initial_seed)

    for iteration in range(NUM_ITERATIONS):
        # Record iteration start time
        iter_start_time = time.time()

        # Force garbage collection
        force_memory_cleanup()

        if args.verbose:
            print(f"Process {accelerator.process_index} starting iteration {iteration + 1}/{NUM_ITERATIONS}")

        # Generate seeds on main process only
        if accelerator.is_main_process:
            if args.verbose:
                print(f"Main process {accelerator.process_index} generating seeds")
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            if args.verbose:
                print(f"Worker process {accelerator.process_index} waiting for seeds")
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        # Broadcast seeds from main process to all processes
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()  # Convert back to list for all processes

        if args.verbose:
            print(f"Process {accelerator.process_index} received seeds")

        # Assign seeds to each process for processing
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            # Simple task assignment: assign seeds by process ID
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))

        if args.verbose:
            print(f"Process {accelerator.process_index} assigned {len(local_seeds)} seeds: {[idx for idx, _ in local_seeds]}")

        # Process seeds in smaller batches to reduce memory pressure
        local_rewards = []
        local_flops = []
        batch_size = max(1, min(args.gpu_threads, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]
            max_workers = 1 if USE_VLLM else len(batch_seeds)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Prepare thread arguments
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    # Pass verbose flag, track_flops, and num_params as arguments to process_seed function
                    thread_args.append((seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, args.verbose, dataset, args.track_flops, num_params))

                # Execute in parallel and collect results
                results = list(executor.map(process_seed, thread_args))
                
                if args.track_flops:
                    for seed_idx, reward, flops in results:
                        local_rewards.append((seed_idx, reward))
                        local_flops.append((seed_idx, flops))
                else:
                    local_rewards.extend(results)

            # Clean up between batches
            force_memory_cleanup()

        # Collect rewards from all processes
        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)

        # Fill in locally computed rewards
        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward

        # Aggregate rewards from all processes (each process will get the full reward list)
        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)

        # Convert aggregated rewards back to Python list
        rewards = all_rewards.cpu().tolist()
        # Clean up no longer needed tensor
        del all_rewards
        
        # Collect FLOPs from all processes if tracking
        iter_flops = 0
        if args.track_flops:
            all_flops = torch.zeros(POPULATION_SIZE, device=accelerator.device, dtype=torch.float64)
            for seed_idx, flops in local_flops:
                all_flops[seed_idx] = flops
            
            if accelerator.num_processes > 1:
                torch.distributed.all_reduce(all_flops, op=torch.distributed.ReduceOp.SUM)
            
            iter_flops = all_flops.sum().item()
            total_flops += iter_flops
            del all_flops
        
        force_memory_cleanup()

        # Convert rewards to a tensor and normalize.
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        if args.verbose:
            print(f"Process {accelerator.process_index} updating model weights")

        # === ES update on trainable params (LoRA params if using LoRA, or all params if not) ===
        original_model = model_list[0]
        param_count = 0
        for name, param in original_model.named_parameters():
            if not param.requires_grad:
                continue
            param_count += 1
            
            # Store original value for debugging
            original_sum = param.data.sum().item() if args.verbose and param_count == 1 else None
            
            if args.verbose and param_count <= 3:  # Log first few params for debugging
                print(f"[DEBUG] Process {accelerator.process_index}: Updating param {name}, shape: {param.shape}")
                print(f"[DEBUG] Original param sum before ES update: {original_sum}")
            
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed))
                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                update.add_(noise.mul(float(r_norm)))
                del noise
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            
            # Log change for first param
            if args.verbose and param_count == 1:
                new_sum = param.data.sum().item()
                update_sum = update.sum().item()
                print(f"[DEBUG] After ES update - param sum: {new_sum}, update sum: {update_sum}, change: {new_sum - original_sum}")
            
            del update
            torch.cuda.empty_cache()
        
        if args.verbose:
            print(f"[DEBUG] Process {accelerator.process_index}: Updated {param_count} trainable parameters")

        # Sync trainable params to sibling thread models
        if args.verbose:
            print(f"[DEBUG] Process {accelerator.process_index}: Syncing updated weights to {len(model_list)-1} sibling models")
        
        for model_idx in range(1, len(model_list)):
            dst = model_list[model_idx]
            # copy by name
            src_named = dict(original_model.named_parameters())
            synced_params = 0
            for n_dst, p_dst in dst.named_parameters():
                if p_dst.requires_grad:
                    p_dst.data.copy_(src_named[n_dst].data)
                    synced_params += 1
            
            if args.verbose:
                print(f"[DEBUG] Process {accelerator.process_index}: Synced {synced_params} params to model {model_idx}")

        # Synchronize to ensure weight updates are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)
        if USE_VLLM:
            if args.verbose:
                print(f"[DEBUG] Process {accelerator.process_index}: Syncing updated weights to vLLM")
            with VLLM_LOCK:
                _sync_hf_to_vllm(original_model)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time

        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()
        std_reward = rewards_tensor.std().item()

        del rewards_tensor, rewards_normalized
        force_memory_cleanup()

        if accelerator.is_main_process:
            flops_str = ""
            if args.track_flops:
                flops_str = f", FLOPs: {iter_flops/1e12:.2f}T"
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}{flops_str}")
            
            # Log to wandb
            if args.use_wandb:
                log_dict = {
                    "iteration": iteration + 1,
                    "reward/mean": mean_reward,
                    "reward/min": min_reward,
                    "reward/max": max_reward,
                    "reward/std": std_reward,
                    "time/iteration_seconds": iter_time,
                }
                if args.track_flops:
                    log_dict["compute/iteration_tflops"] = iter_flops / 1e12
                    log_dict["compute/cumulative_tflops"] = total_flops / 1e12
                    log_dict["compute/tflops_per_second"] = (iter_flops / 1e12) / iter_time
                if torch.cuda.is_available():
                    log_dict["memory/gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
                    log_dict["memory/gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                wandb.log(log_dict)
            
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

            # Save checkpoint every 100 iterations (adapters only)
            if (iteration + 1) % 100 == 0:
                save_model_checkpoint(original_model, tokenizer, iteration + 1, model_name, initial_seed, args, len(dataset))

    total_time = time.time() - training_start_time

    # Save the final model
    if accelerator.is_main_process:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        if args.track_flops:
            print(f"Total FLOPs: {total_flops/1e12:.2f} TFLOPs")
            print(f"Average TFLOPs/s: {(total_flops/1e12)/total_time:.2f}")
        
        if args.use_wandb:
            log_dict = {
                "time/total_training_seconds": total_time, 
                "time/total_training_minutes": total_time/60
            }
            if args.track_flops:
                log_dict["compute/total_tflops"] = total_flops / 1e12
                log_dict["compute/average_tflops_per_second"] = (total_flops / 1e12) / total_time
            wandb.log(log_dict)
        
        question_num = len(dataset)
        save_dir = f"{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_question_num{question_num}_final"
        print(f"Saving final model to {save_dir}...")
        # Save model based on whether LoRA was used
        if args.use_lora:
            # Try to save merged full model; fallback to adapters-only
            try:
                merged = original_model.merge_and_unload()  # returns base model with LoRA merged
                merged.save_pretrained(save_dir)
            except Exception as e:
                print(f"merge_and_unload failed ({e}); saving adapters only.")
                original_model.save_pretrained(save_dir)
        else:
            # Save full model directly
            original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Final model saved successfully.")
        
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()
