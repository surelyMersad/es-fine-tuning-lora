python countdown/es_fine-tuning_countdown.py \
  --use_vllm_rollouts \
  --use_lora \
  --gpu_threads 1 \
  --precision bf16 \
  --hf_cache_dir hf_cache \
  --data_sample 1000 \
  --verbose \
  --use_wandb