import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import copy
import os
import argparse
from accelerate import Accelerator
import wandb
import time
import torch.multiprocessing as mp
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import math
import gc
import json

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='hf_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--gpu_threads', type=int, default=4, help='Number of parallel threads per GPU')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--data_sample', type=int, default=1000, help='Number of data samples to use for training')
args = parser.parse_args()

# Hyperparameters for ES
NUM_ITERATIONS = 1000             # Number of ES iterations (generations)
POPULATION_SIZE = 30              # Population size (number of perturbations per iteration)
SIGMA = 0.001                     # Standard deviation for weight perturbations (noise scale)
ALPHA = 0.0005                    # Learning rate
max_new_tokens = 1024             # Max tokens to generate
do_sample = False                 # Greedy decoding for ES
initial_seed = 33                 # Initial random seed

# Import countdown reward function
from countdown_task import reward_function
print("Using countdown reward function")

def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def save_model_checkpoint(model, tokenizer, iteration, model_name, initial_seed, args, dataset_size):
    """Save model checkpoint at specified iteration"""
    question_num = dataset_size
    save_dir = f"{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{iteration}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_question_num{question_num}_checkpoint"
    print(f"Saving checkpoint at iteration {iteration} to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Checkpoint saved successfully.")

def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False):
    """
    Generate a response from the model given an input (single or batch) and compute rewards.
    """
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    # Handle single or batch input
    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    # Batch tokenization
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
    attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
    with torch.inference_mode():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    # Decode batch outputs
    generated_texts = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)

    del input_ids, outputs
    torch.cuda.empty_cache()

    # Compute rewards
    rewards = []
    for i, (gen_text, tgt_text, inp_text) in enumerate(zip(generated_texts, target_texts, input_texts)):
        numbers = None
        target = None
        if "[" in inp_text and "]" in inp_text:
            start_idx = inp_text.find("[")
            end_idx = inp_text.find("]")
            if start_idx != -1 and end_idx != -1:
                numbers_str = inp_text[start_idx+1:end_idx]
                numbers = [int(n) for n in numbers_str.split() if n.isdigit()]

        if tgt_text.isdigit():
            target = int(tgt_text)

        model_response = gen_text
        if "assistant:" in gen_text:
            model_response = gen_text.split("assistant:")[-1].strip()

        reward_result = reward_function(model_response, numbers, target)
        reward = reward_result["reward"]
        rewards.append(reward)

    if return_text:
        return rewards, generated_texts
    else:
        return rewards

def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose, dataset = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Apply perturbation
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
        param.data.add_(SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # Evaluate all prompts with perturbed weights (batch)
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]
    rewards = evaluate_model(model, tokenizer, input_texts, target_texts, accelerator,
                             seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, return_text=False)
    total_reward = sum(rewards)

    # Restore original weights
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
        param.data.add_(-SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    average_reward = total_reward / len(dataset)
    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")

    return seed_idx, average_reward

# --- Main Evolution Strategies Loop ---
def main():
    # Initialize Accelerate with wandb tracking
    accelerator = Accelerator(log_with="wandb")

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

    if accelerator.is_main_process:
        print(f"Total processes: {accelerator.num_processes}, GPU threads per process: {args.gpu_threads}")
        print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
        print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")

    # ---- Initialize WandB via Accelerate (uses env vars if set) ----
    accelerator.init_trackers(
        project_name=os.getenv("WANDB_PROJECT", "es-finetuning"),
        config={
            "model": args.model_name,
            "precision": args.precision,
            "gpu_threads": args.gpu_threads,
            "data_sample": args.data_sample,
            "algo": "ES",
            "population_size": POPULATION_SIZE,
            "sigma": SIGMA,
            "alpha": ALPHA,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "initial_seed": initial_seed,
        },
    )

    # Load model(s)
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")

    model_list = []
    for model_index in range(args.gpu_threads):
        model_list.append(AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},  # Assign devices explicitly
            torch_dtype=torch.float16 if args.precision == 'fp16' else (torch.bfloat16 if args.precision == 'bf16' else torch.float32),
        ))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)

    if accelerator.is_main_process:
        print("Model loaded successfully")

    for model in model_list:
        model.eval()  # Turn off dropout, etc.

    force_memory_cleanup()

    # Record total training start time
    training_start_time = time.time()

    np.random.seed(initial_seed)

    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        force_memory_cleanup()

        if args.verbose:
            print(f"Process {accelerator.process_index} starting iteration {iteration + 1}/{NUM_ITERATIONS}")

        # Generate/broadcast seeds
        if accelerator.is_main_process:
            if args.verbose:
                print(f"Main process {accelerator.process_index} generating seeds")
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            if args.verbose:
                print(f"Worker process {accelerator.process_index} waiting for seeds")
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        if accelerator.num_processes > 1:
            torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()

        if args.verbose:
            print(f"Process {accelerator.process_index} received seeds")

        # Assign seeds per process
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))

        if args.verbose:
            print(f"Process {accelerator.process_index} assigned {len(local_seeds)} seeds: {[idx for idx, _ in local_seeds]}")

        # Process seeds in batches to reduce memory pressure
        local_rewards = []
        batch_size = max(1, min(args.gpu_threads, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    thread_args.append((seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, args.verbose, dataset))
                results = list(executor.map(process_seed, thread_args))
                local_rewards.extend(results)

            force_memory_cleanup()

        # Collect rewards from all processes
        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)
        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward

        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)

        rewards = all_rewards.cpu().tolist()
        del all_rewards
        force_memory_cleanup()

        # Normalize rewards
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Update weights on the first model, then copy to others
        if args.verbose:
            print(f"Process {accelerator.process_index} updating model weights")
        original_model = model_list[0]
        for name, param in original_model.named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed))
                noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            torch.cuda.empty_cache()

        for model_idx in range(1, len(model_list)):
            original_model_tmp = model_list[model_idx]
            for name, param in original_model_tmp.named_parameters():
                param.data.copy_(original_model.get_parameter(name).data.clone())

        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time
        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()

        # ---- WandB logging (multi-process safe via accelerator.log) ----
        try:
            gpu_alloc_mb = (torch.cuda.memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
            gpu_peak_mb  = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
        except Exception:
            gpu_alloc_mb = gpu_peak_mb = 0.0

        accelerator.log({
            "reward/mean": float(mean_reward),
            "reward/min": float(min_reward),
            "reward/max": float(max_reward),
            "time/iter_s": float(iter_time),
            "system/gpu_mem_alloc_mb": float(gpu_alloc_mb),
            "system/gpu_mem_peak_mb": float(gpu_peak_mb),
            "es/population_size": POPULATION_SIZE,
            "es/sigma": float(SIGMA),
            "es/alpha": float(ALPHA),
            "data/sample_count": len(dataset),
        }, step=iteration + 1)

        if accelerator.is_main_process:
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

            # Save checkpoint every 100 iterations
            if (iteration + 1) % 100 == 0:
                save_model_checkpoint(original_model, tokenizer, iteration + 1, model_name, initial_seed, args, len(dataset))
                # (Optional) log checkpoint as artifact
                ckpt_dir = f"{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{iteration + 1}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_question_num{len(dataset)}_checkpoint"
                try:
                    art = wandb.Artifact(f"{os.path.basename(ckpt_dir)}", type="model")
                    art.add_dir(ckpt_dir)
                    wandb.log_artifact(art)
                except Exception:
                    pass

        del rewards_tensor, rewards_normalized
        force_memory_cleanup()

    total_time = time.time() - training_start_time

    # Save the final fine-tuned model weights.
    if accelerator.is_main_process:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        question_num = len(dataset)
        save_dir = f"{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_question_num{question_num}_final"
        print(f"Saving final model to {save_dir}...")
        original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Final model saved successfully.")
        # (Optional) log final model as artifact
        try:
            art = wandb.Artifact(f"{os.path.basename(save_dir)}", type="model")
            art.add_dir(save_dir)
            wandb.log_artifact(art)
        except Exception:
            pass

    # Clean end for trackers
    accelerator.end_training()

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()
