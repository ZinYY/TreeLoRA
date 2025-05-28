# Pytorch Implementation for "TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree"

This repository contains the official PyTorch implementation of TreeLoRA, an efficient continual learning method for Large Language Models (LLMs) that uses layer-wise LoRA adapters guided by a hierarchical gradient-similarity tree.

## Reference

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{ICML'25:TreeLoRA,
    author = {Yu-Yang Qian and Yuan-Ze Xu and Zhen-Yu Zhang and Peng Zhao and Zhi-Hua Zhou},
    title = {TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree},
    booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
    year = {2025},
    pages = {to appear}
}
```

## Code Structure

```
.
├── data/                    # Data directory for LLM-CL-Benchmark
├── model/                   # Model implementations
│   ├── Regular/            # Regular model implementations
│   │   └── Tree_LoRA.py   # TreeLoRA implementation
│   ├── Dynamic_network/    # Dynamic network implementations
│   └── Replay/            # Replay-based methods
├── training/               # Training related code
│   ├── main.py            # Main training script
│   └── params.py          # Training parameters
├── utils/                  # Utility functions
│   ├── data/              # Data processing utilities
│   ├── flash_attention/   # Flash attention implementation
│   ├── my_peft/          # Custom PEFT implementations
│   └── kd_lora_tree.py   # KD-tree implementation for TreeLoRA
├── inference/             # Inference related code
└── scripts/               # Training and evaluation scripts
```

## Requirements

The main dependencies are listed below. For a complete list, see `requirements.txt`:

```
datasets>=2.8.0
transformers==4.31.0
torch
deepspeed==0.9.0
peft
accelerate==0.15.0
huggingface-hub==0.16.4
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Data and Model Preparation

(i) Extract the dataset in the `data/LLM-CL-Benchmark` directory. The benchmark includes 24 different tasks, including:

- C-STANCE
- NumGLUE-cm
- QQP
- NumGLUE-ds
- MultiRC
- RTE
- yelp
- ScienceQA
- amazon
- MeetingBank
- FOMC
- Lima
- BoolQA
- CB
- Py150
- dbpedia
- WiC
- yahoo
- IMDB
- MNLI
- 20Minuten
- agnews
- COPA
- SST-2

(ii) Download the pre-trained model from HuggingFace and place it in the `./PTM/` directory.

### 3. Training and Evaluating

To train and evaluate a method on the TRACE dataset, just run:

```bash
export model_name="Llama-3.2-1B-Instruct"

# Run training script with default parameters (e.g., TreeLoRA)
bash scripts/lora_based_methods/Tree_LoRA.sh
```

Key parameters in the training script:

-   `--model_name_or_path`: Path to the pretrained model
-   `--data_path`: Path to the training dataset
-   `--dataset_name`: Names of the datasets to train on
-   `--reg`: Regularization parameter (default: 0.5)
-   `--num_train_epochs`: Number of training epochs per task

Or simply, run `./scripts/run_all_exps.sh` to run all the experiments.

## Features

-   Efficient continual learning through layer-wise LoRA adapters
-   Hierarchical gradient-similarity tree for adapter organization
-   Support for multiple LLM architectures (Gemma, LLaMA, Mistral, etc.)
-   DeepSpeed integration for efficient training
-   Flash attention implementation for improved performance
