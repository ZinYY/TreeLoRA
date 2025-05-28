# This is the script to run the tran and evaluate FIX method on the TRACE continual learning benchmark.

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

# FIX need no Train:

# Inference (ICL):
#deepspeed --include=localhost:0 --master_port 25004 inference/ICL.py  \
#    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
#    --dataset_name yahoo,amazon,agnews,dbpedia \
#    --model_name_or_path ./PTM/$model_name \
#    --inference_batch 4 \
#    --max_prompt_len 3584 \
#    --max_ans_len 512 \
#    --seed 1234 \
#    --deepspeed \
#    --demonstrations_num 6 \
#    --inference_output_path ./outputs_LLM-CL/ICL_$now

python inference/infer_multi_command.py  \
    --gpus $gpu_nodes \
    --master_port 25002 \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --inference_tasks yahoo,amazon,agnews,dbpedia \
    --model_name_or_path ./PTM/$model_name \
    --inference_model_path NO_PATH_NEEDED \
    --inference_batch 1 \
    --max_prompt_len 3584 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method FIX \
    --inference_output_path ./outputs_LLM-CL/cl_O_order3/$model_name/FIX_$now/predictions

# Collect results:
python inference/collect_results.py --inference_tasks yahoo,amazon,agnews,dbpedia --data_path ./outputs_LLM-CL/cl_O_order3/$model_name/FIX_$now/predictions
