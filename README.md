# Finetune LLM Math

A comprehensive pipeline for fine-tuning Large Language Models (LLMs) on mathematical tasks, supporting multiple training approaches including Supervised Fine-Tuning (SFT), Proximal Policy Optimization (PPO), and Generalized Reinforcement Policy Optimization (GRPO).

## Features

- Exploratory Data Analysis (EDA) tools for dataset visualization
- Support for multiple training approaches:
  - Supervised Fine-Tuning (SFT)
  - Proximal Policy Optimization (PPO)
  - Generalized Reinforcement Policy Optimization (GRPO)
- Configurable training parameters through YAML configuration
- Hugging Face integration for model and dataset management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/finetune-llm-math.git
cd finetune-llm-math
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Hugging Face token:
```bash
export HF_TOKEN=your_token_here
```

## Usage

The pipeline supports multiple modes of operation:

### Exploratory Data Analysis (EDA)
```bash
python main.py --mode eda
```

### Supervised Fine-Tuning (SFT)
```bash
python main.py --mode sft
```

### Proximal Policy Optimization (PPO)
```bash
python main.py --mode ppo
```

### Generalized Reinforcement Policy Optimization (GRPO)
```bash
python main.py --mode grpo
```

## Configuration

Training parameters can be configured in `configs/config.yaml`. The configuration file includes settings for:
- Model selection
- Dataset configuration
- Training hyperparameters
- Random seed for reproducibility

## Dependencies

- IPython
- PyYAML
- python-dotenv
- Hugging Face libraries (datasets, transformers)
- Matplotlib and Seaborn for visualization
- PyTorch for deep learning