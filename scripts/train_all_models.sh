#!/bin/bash

models=(
    "Llama-3.2-1B-Instruct"
    "Llama-2-7b-chat"
    "Llama-3.1-8B-Instruct"
    "Qwen2.5-7B-Instruct"
    "Mistral-7B-Instruct-v0.3"
    "gemma-2b-it"
)

timestamp=$(date +"%Y%m%d_%H%M%S")

for model in "${models[@]}"; do
    export model_name=$model
    echo "start training: $model_name"
    bash scripts/lora_based_methods/Tree_LoRA.sh > "train_${model}_${timestamp}.log" 2>&1
done 