export model_name="Llama-2-7b-chat"
#model_name="Llama-3.2-1B-Instruct"
#export model_name="Mistral-7B-Instruct-v0.3"
#model_name="gemma-2b-it"

export dataset_name="dbpedia,amazon,yahoo,agnews"

timestamp=$(date +"%Y%m%d_%H%M%S")


chmod +x ./scripts/full_grad_methods/FIX.sh
sed -i 's/\r$//' ./scripts/full_grad_methods/FIX.sh
./scripts/full_grad_methods/FIX.sh > shell_outputs/FIX_${timestamp}_${model_name}.log 2>&1

chmod +x ./scripts/lora_based_methods/lora.sh
sed -i 's/\r$//' ./scripts/lora_based_methods/lora.sh
./scripts/lora_based_methods/lora.sh > shell_outputs/lora_${timestamp}_${model_name}.log 2>&1

chmod +x ./scripts/full_grad_methods/base.sh
sed -i 's/\r$//' ./scripts/full_grad_methods/base.sh
./scripts/full_grad_methods/base.sh > shell_outputs/base_${timestamp}_${model_name}.log 2>&1

chmod +x ./scripts/full_grad_methods/GEM.sh
sed -i 's/\r$//' ./scripts/full_grad_methods/GEM.sh
./scripts/full_grad_methods/GEM.sh > shell_outputs/GEM_${timestamp}_${model_name}.log 2>&1

chmod +x ./scripts/full_grad_methods/EWC.sh
sed -i 's/\r$//' ./scripts/full_grad_methods/EWC.sh
./scripts/full_grad_methods/EWC.sh > shell_outputs/EWC_${timestamp}_${model_name}.log 2>&1

chmod +x ./scripts/lora_based_methods/hidelora.sh
sed -i 's/\r$//' ./scripts/lora_based_methods/hidelora.sh
./scripts/lora_based_methods/hidelora.sh > shell_outputs/hidelora_${timestamp}_${model_name}.log 2>&1

chmod +x ./scripts/lora_based_methods/O_LoRA.sh
sed -i 's/\r$//' ./scripts/lora_based_methods/O_LoRA.sh
./scripts/lora_based_methods/O_LoRA.sh > shell_outputs/O_LoRA_${timestamp}_${model_name}.log 2>&1

chmod +x ./scripts/lora_based_methods/Tree_LoRA.sh
sed -i 's/\r$//' ./scripts/lora_based_methods/Tree_LoRA.sh
./scripts/lora_based_methods/Tree_LoRA.sh > shell_outputs/Tree_LoRA_${timestamp}_${model_name}.log 2>&1