#!/bin/bash

models=(
    "Llama-3.2-1B-Instruct"
    "Llama-2-7b-chat"
#    "Llama-3.1-8B-Instruct"
#    "Qwen2.5-7B-Instruct"
    "Mistral-7B-Instruct-v0.3"
    "gemma-2b-it"
)

epochs_settings=(
    "2,1,3,2,1,2,2,3"
    "5,3,7,5,3,5,5,7"
)

timestamp=$(date +"%Y%m%d_%H%M%S")

gpu_nodes="0,1,2,3"

reg=0.5

for model in "${models[@]}"; do
    for epochs in "${epochs_settings[@]}"; do
        model_name=$model
        now=$(date +"%m%d_%H%M%S")
        echo "start training: $model_name (epochs: $epochs)"

        echo "Start training..."
        deepspeed --include=localhost:$gpu_nodes --master_port 25011 training/main.py \
            --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
            --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
            --model_name_or_path ./PTM/$model_name \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --max_prompt_len 1024 \
            --max_ans_len 512 \
            --learning_rate 1e-4 \
            --weight_decay 0. \
            --num_train_epochs $epochs \
            --gradient_accumulation_steps 8 \
            --lr_scheduler_type cosine \
            --num_warmup_steps 0 \
            --seed 1234 \
            --zero_stage 2 \
            --deepspeed \
            --print_loss \
            --CL_method Tree_LoRA \
            --output_dir ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_${now}_epochs${epochs} \
            --reg $reg \
            > "./shell_outputs/${model}_epochs${epochs}_${timestamp}.log" 2>&1

        echo "Start inference..."
        python inference/infer_multi_command.py \
            --gpus $gpu_nodes \
            --master_port 25011 \
            --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
            --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
            --model_name_or_path ./PTM/$model_name \
            --inference_model_path ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_${now}_epochs${epochs} \
            --inference_batch 1 \
            --max_prompt_len 1024 \
            --max_ans_len 512 \
            --seed 1234 \
            --CL_method Tree_LoRA \
            --inference_output_path ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_${now}_epochs${epochs}/predictions

        echo "Start collecting results..."
        python inference/collect_results.py --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten --data_path ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_${now}_epochs${epochs}/predictions

        sleep 30
    done
done