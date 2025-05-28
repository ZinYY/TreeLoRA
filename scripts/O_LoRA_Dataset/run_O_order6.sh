#export model_name="Llama-2-7b-chat"
#export model_name="Llama-2-7b-hf"
export model_name="Llama-3.2-1B-Instruct"
#export model_name="Mistral-7B-Instruct-v0.3"
#export model_name="gemma-2b-it"
#export model_name="Qwen2.5-32B-Instruct"

export model_name="Llama-3.2-1B-Instruct"
./scripts/rebuttal_O_LoRA_long/order3/lora_based_methods/lora.sh

export model_name="Llama-3.2-1B-Instruct"
./scripts/rebuttal_O_LoRA_long/order3/lora_based_methods/O_LoRA.sh

export model_name="Llama-3.2-1B-Instruct"
./scripts/rebuttal_O_LoRA_long/order3/lora_based_methods/hidelora.sh

export model_name="Llama-3.2-1B-Instruct"
./scripts/rebuttal_O_LoRA_long/order3/lora_based_methods/Tree_LoRA.sh

export model_name="Llama-3.2-1B-Instruct"
./scripts/rebuttal_O_LoRA_long/order3/full_grad_methods/FIX.sh



./scripts/rebuttal_O_LoRA_long/order3/full_grad_methods/base.sh

./scripts/rebuttal_O_LoRA_long/order3/full_grad_methods/GEM.sh

./scripts/rebuttal_O_LoRA_long/order3/full_grad_methods/EWC.sh


