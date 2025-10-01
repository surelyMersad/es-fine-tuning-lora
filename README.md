# es-fine-tuning-paper
This repo contains the source code for the paper "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning". Evolution strategies (ES) is used to directly optimize billions of parameters of large language models (LLMs).

Note: we are still actively adding more experimental codes into this repo.

## Setup
Create a virtual environment with python version >= 3.10 and activate it
```bash
python -m venv es
source es/bin/activate
```

From the root of the repository run following command to install all the relevant python packages
```bash
pip install -r requirement.txt
```

## Usage
For running the main ES code on conciseness fine-tuning

```bash
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --machine_rank 0 \
    es_fine-tuning_conciseness.py \
    --gpu_threads=2 \
    --model_name=Qwen/Qwen2.5-7B-Instruct
```

`--num_processes` specifies the number of GPUs to use and `--gpu_threads` specifies the number of threads inside each GPU. The total number of parallel evaluations is thereby equal to `num_processes`*`gpu_threads`.

For running the main ES code on countdown task
```bash
accelerate launch \
    --num_processes 4 \
    --num_machines 1\
    --machine_rank 0 \
    countdown/es_fine-tuning_countdown.py \
    --data_sample 200 \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --gpu_threads 2 \
```

## Other Parameters

- `--gpu_ids`: Specify which GPUs to use (CUDA device id)
- `--model_name`: HuggingFace model to fine-tune
- `--hf_cache_dir`: Directory for HuggingFace cache
- `--precision`: Model precision, default to be `bf16`
- `--verbose`: Enable detailed logging if this argument is present in the command line


## Citation

If you find this work helpful in your research, please cite:

```bibtex
@misc{qiu2025evolutionstrategiesscalellm,
      title={Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning}, 
      author={Xin Qiu and Yulu Gan and Conor F. Hayes and Qiyao Liang and Elliot Meyerson and Babak Hodjat and Risto Miikkulainen},
      year={2025},
      eprint={2509.24372},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.24372}, 
}
```
