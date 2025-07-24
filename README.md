# AutoRule

Official repository for **AutoRule: Reasoning Chain-of-thought Extracted Rule-based Rewards Improve Preference Learning**. [Arxiv](https://arxiv.org/abs/2506.15651)

![AutoRule Overview](images/AutoRule-overview.png)

## Setup

Install dependencies for dataset preprocessing and rule extraction:
```bash
pip install -r requirements.txt
```

Add your HuggingFace token to download the required UltraFeedback-binarized and MT-Bench Human Judgement datasets:
```bash
huggingface-cli login
```

To extract rules with our scripts, you also need access to Amazon Bedrock. Set the following environment variables:
```bash
export AWS_ACCESS_KEY_ID=<YOUR_KEY>
export AWS_SECRET_ACCESS_KEY=<YOUR_KEY>
```

To train AutoRule models, install dependencies for OpenRLHF by following the instructions in `src/OpenRLHF/readme.md`. You'll also need to set `WANDB_TOKEN` as an environment variable for training logs/metrics. To obtain a token, go to https://wandb.ai/authorize.

## Dataset Preprocessing

To prepare the dataset in the correct format for reward model training, run:
```bash
python src/preprocess_ultrafeedback.py --hf_username <your_username>
```

**Arguments:**
- `hf_username` - Your Hugging Face username to upload the processed dataset

## AutoRule Extraction

Extract rules from preference data using:
```bash
python src/autorule.py --dataset_type <uf|mt> --output_dir <output_directory>
```

**Arguments:**
- `dataset_type` - Choose `uf` (UltraFeedback) or `mt` (MT-Bench)
- `output_dir` - Directory to save the extracted rules
- `uf_num_examples` - Number of examples for rule extraction from UF dataset (default: 256)
- `mt_num_examples_per_question` - Number of examples per MT query for rule extraction (default: 8)
- `num_proc` - Number of processes for parallel evaluation (default: 8)

## Training

We use [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) as our training framework with additional modifications to support AutoRule experiments.

Training scripts are designed for SLURM jobs on the CMU LTI Babel cluster, but can be adapted for other environments.

Scripts are located in their own directory:
```bash
cd scripts
```

### SFT (Supervised Fine-tuning)

Edit `train_llama_slurm.sh` to the following:
```bash
readonly training_script="train_sft_llama_ultrafeedback.sh" 
```
Then:
```bash
sbatch train_llama_slurm.sh
```

### Reward Model Training

Edit `train_llama_slurm.sh` to the following:
```bash
readonly training_script="train_rm_llama_ultrafeedback.sh" 
```

Then:
```bash
sbatch train_llama_slurm.sh
```

### RL Stage

Perform reinforcement learning with various reward configurations:

#### RLHF Training with PPO
```bash
sbatch train_ppo_llama_ray_slurm.sh
```


#### Standard GRPO Training
```bash
sbatch train_grpo_llama_ray_baseline_slurm.sh
```

#### Other GRPO baselines
Length control:
```bash
sbatch train_grpo_llama_ray_length_control_slurm.sh
```

Length penalty:
```bash
sbatch train_grpo_llama_ray_length_penalty_slurm.sh
```

#### AutoRule Training
For UltraFeedback rules:
```bash
sbatch train_grpo_llama_ray_autorule_slurm_uf.sh
```

For MT-Bench rules:
```bash
sbatch train_grpo_llama_ray_autorule_slurm_mt.sh
```

## Evaluation

We evaluated AutoRule against baselines using three evaluation benchmarks:

### UltraFeedback Win Rate

To select data for testing, we filter the `test_prefs` split and only include the examples where the chosen and rejected responses are both less than 512 tokens, the chosen score is higher than the rejected score, and the word ``confidence" is not in the either response (as done in https://github.com/PorUna-byte/PAR). We also only select a maximum amount of 1024 examples.

The prompt used for UltraFeedback win rate is below (`instruction`, `output_1`, and `output_2` are inputs):
```
I want you to create a leaderboard of different large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{{
    "instruction": """{instruction}"""
}}

Here are the outputs of the models:
[
    {{
        "model": "model_1",
        "answer": """{output_1}"""
    }},
    {{
        "model": "model_2",
        "answer": """{output_2}"""
    }}
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {{'model': <model-name>, 'rank': <model-rank>}},
    {{'model': <model-name>, 'rank': <model-rank>}}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give.
```

### AlpacaEval 2.0

We used the following config for all evaluations on AlpacaEval 2.0:
```yaml
[model-name]:
  prompt_template: "[model-name]/prompt.txt"
  fn_completions: "vllm_local_completions"
  completions_kwargs:
    model_name: "[model-name/path]"
    batch_size: 16
    max_new_tokens: 4096
    model_kwargs:
      dtype: bfloat16
    temperature: 0.9
    top_p: 1.0
```

`prompt.txt` below:
```
<|user|>{instruction}<|assistant|>
```


### MT-Bench

Use the following adapter in `FastChat/fastchat/model/model_adapter.py`:
```python
class ARLlamaAdapter(BaseModelAdapter):
    """The model adapter for AutoRule and its baselines"""

    def match(self, model_path: str):
        return "ar-llama" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ar-llama")
```
And register it in the same file:
```python
register_model_adapter(ARLlamaAdapter)
```

Additionally, add the following conversation template to `FastChat/fastchat/conversation.py`:
```python
register_conv_template(
    Conversation(
        name="ar-llama",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
    )
)
```