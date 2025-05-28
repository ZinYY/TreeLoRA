export model_name="Llama-2-7b-chat"
#export dataset_name="agnews,dbpedia,yelp,yahoo,amazon"
export dataset_name="dbpedia,amazon,yahoo,agnews"

now=$(date +"%m%d_%H%M%S")
timestamp=$now
#get GPUs:
gpu_nodes="0,1,2,3"

#epochs=1,1,5,5,1,5,5,5
epochs=2,1,3,2,1,2,2,3
#epochs=5,3,7,5,3,5,5,7

reg=0.5

epochs=5,3,7,5,3,5,5,7

lr=1e-4

# Train:
deepspeed --include=localhost:$gpu_nodes --master_port 25007 training/main.py  \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name $dataset_name \
    --model_name_or_path ./PTM/$model_name \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate $lr \
    --weight_decay 0. \
    --num_train_epochs $epochs \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method O_LoRA \
    --output_dir ./outputs_LLM-CL/cl/$model_name/O_LoRA_$now \
    2>&1 | tee "./shell_outputs/${model_name}_O_LoRA_${timestamp}.log"


# Inference:
python inference/infer_multi_command.py  \
    --gpus $gpu_nodes \
    --master_port 25007 \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --inference_tasks $dataset_name \
    --model_name_or_path ./PTM/$model_name \
    --inference_model_path ./outputs_LLM-CL/cl/$model_name/O_LoRA_$now \
    --inference_batch 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method O_LoRA \
    --inference_output_path ./outputs_LLM-CL/cl/$model_name/O_LoRA_$now/predictions

# Collect results:
python inference/collect_results.py --inference_tasks dbpedia,amazon,yahoo,agnews --data_path ./outputs_LLM-CL/cl/$model_name/O_LoRA_$now/predictions



