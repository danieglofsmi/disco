# Eliciting Diverse Thinking Schemata for Large Reasoning Models
## Usage
### Installtion
```
# Create conda environment
conda create -n disco-env python=3.11
conda activate disco-env

# Install dependencies
cd disco-env
pip install -r requirements.txt
pip install -e .

# Install llama-factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation

# Install veRL framework, recommended version: 0.4.0.dev0
git clone https://github.com/volcengine/verl && cd verl
pip3 install --no-deps -e .

# Install Flash Attention (recommended version: 2.7.4.post1)
pip install flash-attn --no-build-isolation

# Install vllm (recommended version: 0.8.3)
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .

```

### Repository Structure
This repository includes:
- sample_data: Data samples for training DiScO models.
  - thread_api-label.py: Code for labeling sft data.
  - preprocess_deepscaler.py: Code for preprocessing DeepScaler dataset.
- sft.yaml: Example scripts to train DiScO with llama-factory.
- data_process.py: Used for preprocessing data prior to grpo training.
- rewards.py: The rewards used in GRPO training, can be used directly to replace Python file in the verl directory.
- grpo_train.sh: Example scripts to train DiScO with verl.
- inference_vllm.py: Code for vllm inference and evaluating DiScO models.
- verify.py: Code for verify the answer correctness in reasoning.
- process_null.py: Process samples with incomplete reasoning (where the answer is null) and perform Initial Truncation or Truncation with Repetition Elimination to enable continued reasoning completion.

### Datasets
Links to the dataset used:
#### Training
- OpenR1: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- DeepScaler: https://huggingface.co/datasets/lime-nlp/DeepScaleR_Difficulty
#### Evaluation
- AIME 2024: https://huggingface.co/datasets/math-ai/aime24
- AIME 2025: https://huggingface.co/datasets/math-ai/aime25
- AMC 2023: https://huggingface.co/datasets/math-ai/amc23
- MATH-500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- GSM8k: https://huggingface.co/datasets/openai/gsm8k
- GPQA-Diamond: https://github.com/idavidrein/gpqa

### Base models
DeepSeek-R1-Distill-Qwen-7B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DeepSeek-R1-Distill-Qwen-32B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B