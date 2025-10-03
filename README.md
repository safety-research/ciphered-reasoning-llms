# All Code, No Thought: Language Models Struggle to Reason in Ciphered Language

This repo contains the code used to run all experiments for the [paper](https://cipheredreasoning.app/).

## Setup

To run the example scripts for fine-tuning & few-shot prompting models to reason in ciphered language and translate ciphered language, you will need at least an 8xA100 machine. We use [Ray](https://www.ray.io/) to orchestrate experiments. Jobs are defined as YAML files containing experiment parameters and a set of stages, and JSON files containing combinations of parameters to grid over.

You will need to run the script in `orchestration/base_setup.sh` to install all dependencies and create a virtual environment. Please ensure the following:
- The script assumes you have root access on an Ubuntu machine; you may need to modify it if you have a different setup.
- The script assumes you have cloned the `jeff-encoding-schemes` branch of [verl](git@github.com:safety-research/verl.git) to `~/sky_workdir/verl`. Please modify it accordingly if your path is different. We have added additional training features & infrastructure optimization to the SFT training script in verl for our use here.
- We also use Supabase for tracking experiment metadata, so you should set `SUPABASE_CONNECTION_URL` appropriately.
- In `env/anthropic.py` and `env/openai.py`, you should create files with `set_anthropic_key` and `set_openai_key` Python functions that set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` environment variables appropriately.

Some Jupyter notebooks assume that this repo is cloned into `/home/ubuntu/sky_workdir/ciphered-reasoning-llms`. You may need to adjust `sys.path.append` calls appropriately to reflect your path. Scripts have only been tested on H100 machines and therefore CUDA/vllm/PyTorch versions may require adjustment for your setup.

## Running experiments

Running these scripts will execute the entire workflow, from datagen to fine-tuning to inference to evaluation. Each experiment has a hash generated from its parameters. All model weights, generated outputs, and datasets will be generated into this folder. The hash will be printed to stdout.

```
# Assuming you are in the venv:

# Fine-tune Qwen2.5-7B-Instruct to reason in base64
python orchestration/grid_results.py examples/sft.json --eval-script examples/sft.yaml

# Few-shot prompt Sonnet 4 to reason in base64
python orchestration/grid_results.py examples/fewshot.json --eval-script examples/fewshot.yaml

# Generally prompted zero shot translation
python orchestration/grid_results.py examples/generally_prompted_translation.json --eval-script examples/generally_prompted_translation.yaml
```

For some few-shot results, we used additional instructions in the user prompt or system prompt to improve adherence. You may grep for JSON files containing the string `user_prompt_suffix_override` or `system_prompt_override` to find these prompts.

## Pretraining prevalence

We compute pretraining prevalence using the notebook in `experiments/20250811/compute_pretraining_prevalence.ipynb`. 
