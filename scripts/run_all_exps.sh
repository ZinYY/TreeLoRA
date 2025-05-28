#export model_name="Llama-2-7b-chat"
#export model_name="Llama-2-13b-chat-hf"
export model_name="Llama-3.2-1B-Instruct"
#export model_name="Mistral-7B-Instruct-v0.3"
#export model_name="gemma-2b-it"
#export model_name="Qwen2.5-32B-Instruct"

#export model_name="Llama-2-13b-chat-hf"
./scripts/full_grad_methods/FIX.sh

./scripts/lora_based_methods/lora.sh

./scripts/full_grad_methods/base.sh

./scripts/full_grad_methods/GEM.sh

./scripts/full_grad_methods/EWC.sh

./scripts/lora_based_methods/hidelora.sh

./scripts/lora_based_methods/O_LoRA.sh

./scripts/lora_based_methods/Tree_LoRA.sh
