"""
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
"""

# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import random

import math
import sys
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import deepspeed
import json

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)

import deepspeed

from ICL import TASK_PROMT, Constrained_PROMPT

# from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten  # to be continued
from training.params import Method2Class, AllDatasetName

from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.DualPrompt import convert_DualPrompt_model


# dist.init_process_group(backend='nccl')


# add flash attention
from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn

replace_llama_attn_with_flash_attn()
replace_bloom_attn_with_flash_attn()

def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset. A single data path.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='./tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--inference_model_path",
        type=str,
        help=
        "Path to inference model.",
        required=True,
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    # inference params
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=256,
        help="The maximum answer length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generate temperature params.",
    )
    
    parser.add_argument(
        "--lora_depth",
        type=int,
        default=-1,
        help="max depth of lora. -1 means no limit.",
    )
    parser.add_argument(
        "--inference_batch",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    #  add other inference params
    parser.add_argument(
        "--inference_tasks",
        type=list_of_strings,
        default='all',
        help='Datasets to be used.'
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    
    
    parser.add_argument('--inference_output_path',
                        type=str,
                        default=None,
                        help="Where to store inference results.")
    parser.add_argument('--CL_method',
                        default=None,
                        help='continual learning method used')
    
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--current_rank", type=int, required=True)
    parser.add_argument("--total_rank", type=int, required=True)
    parser.add_argument("--output_json_file_path", type=str, required=True)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    return args


def get_random_demonstrations(dem_num, infer_dataset, length_limit, task, tokenizer):
    length_limit_per_sample = length_limit / (dem_num * 2)
    demonstrations = []
    answers = []
    i = 0
    round = 0
    while i < dem_num:
        round += 1
        if round == 10000:
            break
        demonstration_id = random.randint(0, len(infer_dataset) - 1)
        demonstration = infer_dataset[demonstration_id]  # [{prompt*4},{answer*4}]
        if task != "Py150":
            demonstration["prompt"] = demonstration["prompt"][len(TASK_PROMT[task]):]
        if len(tokenizer(demonstration["prompt"])['input_ids']) + len(tokenizer(demonstration["answer"])['input_ids']) <= length_limit_per_sample:
            if task == "FOMC" or task == "C-STANCE":
                if answers.count(demonstration["answer"]) < dem_num / 3:
                    demonstrations.append(demonstration)
                    answers.append(demonstration["answer"])
                    i += 1
            else:
                if demonstration["answer"] not in answers:
                    demonstrations.append(demonstration)
                    answers.append(demonstration["answer"])
                    i += 1
        else:
            continue
        
        if len(demonstrations) == dem_num:
            return demonstrations
    
    return demonstrations


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda")
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    
    def prediction(model, infer_dataloader):
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        model.eval()
        for step, batch in enumerate(infer_dataloader):
            #  add prompts, choosen, rejected
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            sources_sequences += batch['sources']
            ground_truths += batch['gts']
            del batch['sources']
            del batch['gts']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]
            # update progress bar
            progress_bar.update(1)
            description = f"Step {step}"
            progress_bar.set_description(description, refresh=False)
            with torch.no_grad():
                #  add more inference params
                # backbone config
                # generate_ids = model.generate(batch['input_ids'], max_new_tokens=args.max_ans_len,
                #                               pad_token_id=tokenizer.eos_token_id, attention_mask = batch['attention_mask'], temperature=0.7, do_sample=True, repetition_penalty=2.0 )
                # sft config
                generate_ids = model.generate(input_ids=batch['input_ids'],
                                              attention_mask=batch['attention_mask'],
                                              max_new_tokens=args.max_ans_len,
                                              bos_token_id=tokenizer.bos_token_id,
                                              eos_token_id=tokenizer.eos_token_id,
                                              pad_token_id=tokenizer.unk_token_id,
                                              temperature=args.temperature,
                                              do_sample=True,
                                              num_return_sequences=1,
                                              use_cache=True
                                              )
            sequences = tokenizer.batch_decode(generate_ids[:, prompt_len:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
            predicted_sequences += sequences
        return sources_sequences, predicted_sequences, ground_truths
    
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"
    
    # set evaluation batch size
    # only support bs = 1, cause right padding training logic
    # modify left pad for training and inference
    inference_tasks = args.inference_tasks
    task_num = len(inference_tasks)
    
    round = args.round
    
    inference_model_path = os.path.join(args.inference_model_path, str(round))
    # print_rank_0("Inference Model Path: " + inference_model_path, args.local_rank)
    
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=None,
                            )
    
    if args.CL_method == "FIX":
        pass
    
    # add adapters
    if args.CL_method == "LFPT5":
        from utils.my_peft import PeftModel
        model = PeftModel.from_pretrained(model, inference_model_path)
    
    if args.CL_method == "O_LoRA" or args.CL_method == "Hide_LoRA":
        from utils.my_peft import PeftModel
        model = PeftModel.from_pretrained(model, inference_model_path)
        for name, param in model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = False
            elif name.find("lora_") != -1:
                param.requires_grad = False
    
    if args.CL_method == "Tree_LoRA":
        from utils.my_peft import PeftModel
        model = PeftModel.from_pretrained(model, inference_model_path)
        for name, param in model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = False
            elif name.find("lora_") != -1:
                param.requires_grad = False


    if args.CL_method == "TaSL":
        from utils.my_peft import PeftModel
        # For TaSL, use the consolidated model path which has merged parameters
        consolidated_path = os.path.join(args.inference_model_path, f"{round}_consolidated")
        if os.path.exists(consolidated_path):
            print(f"Using TaSL consolidated model from {consolidated_path}")
            model = PeftModel.from_pretrained(model, consolidated_path)
        else:
            # Fall back to regular path if consolidated not found
            print(f"Consolidated model not found, using regular model from {inference_model_path}")
            model = PeftModel.from_pretrained(model, inference_model_path)

        # Set all parameters to non-trainable for inference
        for name, param in model.named_parameters():
            if "lora_" in name or "loranew_" in name:
                param.requires_grad = False
    
    if args.CL_method == "SAPT":
        from utils.my_peft import PeftModel
        model = PeftModel.from_pretrained(model, inference_model_path)
        for name, param in model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = False
            elif name.find("lora_") != -1:
                param.requires_grad = False

    if args.CL_method == "OGD":
        # from utils.my_peft import get_peft_model, LoraConfig, TaskType
        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        # )
        # model = get_peft_model(model, peft_config)
        from utils.my_peft import PeftModel
        model = PeftModel.from_pretrained(model, inference_model_path)
        for name, param in model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = False
            elif name.find("lora_") != -1:
                param.requires_grad = False
    
    if args.CL_method == "lora":
        # from utils.my_peft import PeftModel
        # model = PeftModel.from_pretrained(model, inference_model_path)
        
        from utils.my_peft import PeftModel
        model = PeftModel.from_pretrained(model, inference_model_path)
        for name, param in model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = False
            elif name.find("lora_") != -1:
                param.requires_grad = False
    
    if args.CL_method == "DualPrompt":
        if "opt" in args.model_name_or_path.lower():
            embed_tokens_shape = model.model.decoder.embed_tokens.weight.shape
            embed_tokens = model.model.decoder.embed_tokens
            args.embed_tokens_dim = embed_tokens_shape[1]
            args.embed_tokens_length = embed_tokens_shape[0]
            args.embed_tokens = embed_tokens
        elif "llama" in args.model_name_or_path.lower():
            embed_tokens_shape = model.model.embed_tokens.weight.shape
            embed_tokens = model.model.embed_tokens
            args.embed_tokens_dim = embed_tokens_shape[1]
            args.embed_tokens_length = embed_tokens_shape[0]
            args.embed_tokens = embed_tokens
        
        elif "qwen" in args.model_name_or_path.lower():
            embed_tokens_shape = model.model.embed_tokens.weight.shape
            embed_tokens = model.model.embed_tokens
            args.embed_tokens_dim = embed_tokens_shape[1]
            args.embed_tokens_length = embed_tokens_shape[0]
            args.embed_tokens = embed_tokens
        
        elif "mistral" in args.model_name_or_path.lower():
            embed_tokens_shape = model.model.embed_tokens.weight.shape
            embed_tokens = model.model.embed_tokens
            args.embed_tokens_dim = embed_tokens_shape[1]
            args.embed_tokens_length = embed_tokens_shape[0]
            args.embed_tokens = embed_tokens
        else:
            embed_tokens_shape = model.model.embed_tokens.weight.shape
            embed_tokens = model.model.embed_tokens
            args.embed_tokens_dim = embed_tokens_shape[1]
            args.embed_tokens_length = embed_tokens_shape[0]
            args.embed_tokens = embed_tokens
        
        if args.CL_method == "DualPrompt":
            args.train_task_list = args.inference_tasks
            args.pool_size = 10
            args.g_prompt_length = 2
            args.e_prompt_length = 3
            args.prompt_init = "uniform"
            model = convert_DualPrompt_model(model, args)
            for name, params in model.named_parameters():
                if "prompt" not in name:
                    params.requires_grad = False
        
    
    # lora_methods: [LFPT5, lora, O_LoRA, Tree_LoRA, OGD]
    # lora not in lower case:
    # if "lora" not in str(args.CL_method).lower() and args.CL_method not in ['LFPT5', 'OGD', 'FIX']:
    if "lora" not in str(args.CL_method).lower() and args.CL_method not in ['LFPT5', 'OGD', 'FIX', 'TaSL', 'SAPT']:
        print("Line 400")
        # if args.CL_method != "O_LoRA" and args.CL_method != "Tree_LoRA" and args.CL_method != "LFPT5":
        inference_model = torch.load(os.path.join(inference_model_path, "pytorch_model.bin"))
        for name, param in model.named_parameters():
            param.data.copy_(inference_model[name])
        del inference_model
    
    if '32b' in str(args.model_name_or_path).lower():
        # convert model to bf16:
        print("\033[34m" + "Convert model to bf16" + "\033[0m")
        model.to(dtype=torch.bfloat16, device=device)
    else:
        model.to(device)
    
    all_results_dic = {}
    for inference_task_id in range(round + 1):  # evaluation for previous tasks in a single round
        inference_task = inference_tasks[inference_task_id]
        dataset_path = os.path.join(args.data_path, inference_task)
        print("\033[34m" + "Start inference Task {}, {}".format(inference_task_id, inference_task) + "\033[0m")
        # Prepare the data
        _, _, infer_dataset = create_prompt_dataset(
            args.local_rank,
            dataset_path,
            args.data_output_path,
            args.seed,
            distributed=False
        )
        if args.CL_method != 'FIX':
            inf_data_collator = DataCollator(
                tokenizer,
                model=model,
                padding="longest",
                max_prompt_len=args.max_prompt_len,
                max_ans_len=args.max_ans_len,
                pad_to_multiple_of=8,
                inference=True
            )
        else:
            # For FIX: ICL!
            args.demonstrations_num = 6
            demonstrations = get_random_demonstrations(
                int(args.demonstrations_num), infer_dataset,
                args.max_prompt_len - len(tokenizer(TASK_PROMT[inference_task] + Constrained_PROMPT)['input_ids']),
                inference_task, tokenizer)
            # print_rank_0("demonstrations length:{}".format(len(demonstrations)), args.global_rank)
            print("demonstrations length:{}".format(len(demonstrations)))
            if inference_task == "MeetingBank":
                demonstrations = []
            inf_data_collator = DataCollator(
                tokenizer,
                model=model,
                padding="longest",
                max_prompt_len=args.max_prompt_len,
                max_ans_len=args.max_ans_len,
                pad_to_multiple_of=8,
                inference=True,
                demonstrations=demonstrations,
                task=inference_task
            )
            
        # sample only part of the infer_dataset
        current_rank = args.current_rank
        total_rank = args.total_rank
        
        part_length = len(infer_dataset) // total_rank
        
        # # DEBUG:
        # part_length = len(infer_dataset) // total_rank // 10
        
        if current_rank < total_rank:
            subset_indices = list(range(current_rank * part_length, (current_rank + 1) * part_length))
        else:
            subset_indices = list(range(current_rank * part_length, len(infer_dataset)))
        
        subset = torch.utils.data.Subset(infer_dataset, subset_indices)
        
        infer_sampler = SequentialSampler(subset)
        infer_dataloader = DataLoader(subset,
                                      collate_fn=inf_data_collator,
                                      sampler=infer_sampler,
                                      batch_size=args.inference_batch)
        
        progress_bar = tqdm(total=len(infer_dataloader), leave=True)
        
        # Inference !
        # print_rank_0("***** Start inference *****", args.local_rank)
        # print red:
        sources_sequences, predicted_sequences, ground_truths = prediction(model, infer_dataloader)
        
        all_results_dic[inference_task] = {}
        all_results_dic[inference_task]["sources_sequences"] = str(sources_sequences)
        all_results_dic[inference_task]["predicted_sequences"] = str(predicted_sequences)
        all_results_dic[inference_task]["ground_truths"] = str(ground_truths)
    # print(all_results_dic)
    
    # save all_results_dic to args.output_json_file_path:
    with open(args.output_json_file_path, 'w') as f:
        # write str(all_results_dic) to file
        f.write(str(all_results_dic))


if __name__ == "__main__":
    main()
