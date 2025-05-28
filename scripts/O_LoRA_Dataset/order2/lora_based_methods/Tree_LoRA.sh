# This is the script to run the tran and evaluate Tree_LoRA method on the TRACE continual learning benchmark.

#get current time:
now=$(date +"%m%d_%H%M%S")
#get GPUs:
gpu_nodes="0,1,2,3"

#model_name="Llama-2-7b-chat"
#model_name="Llama-3.1-8B-Instruct"
#model_name="Llama-3.2-1B-Instruct"
#model_name="Qwen2.5-7B-Instruct"
#model_name="Mistral-7B-Instruct-v0.3"
#model_name="gemma-2b-it"

#epochs=1,1,1,1
epochs=1,1,1,1
#epochs=1,1,1,1

reg=0.5

# Train:
echo "Start training..."
deepspeed --include=localhost:$gpu_nodes --master_port 25011 training/main.py  \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name dbpedia,amazon,agnews,yahoo \
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
    --output_dir ./outputs_LLM-CL/cl_O_order2/$model_name/Tree_LoRA_$now \
    --reg $reg


# Inference:
echo "Start inference..."
python inference/infer_multi_command.py  \
    --gpus $gpu_nodes \
    --master_port 25011 \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --inference_tasks dbpedia,amazon,agnews,yahoo \
    --model_name_or_path ./PTM/$model_name \
    --inference_model_path ./outputs_LLM-CL/cl_O_order2/$model_name/Tree_LoRA_$now \
    --inference_batch 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method Tree_LoRA \
    --inference_output_path ./outputs_LLM-CL/cl_O_order2/$model_name/Tree_LoRA_$now/predictions

# Collect results:
echo "Start collecting results..."
python inference/collect_results.py --inference_tasks dbpedia,amazon,agnews,yahoo --data_path ./outputs_LLM-CL/cl_O_order2/$model_name/Tree_LoRA_$now/predictions
