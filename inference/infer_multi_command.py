# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import copy
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import math
import sys
from tqdm import tqdm

import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten, eval_amazon, eval_yelp, eval_agnews, eval_dbpedia, eval_yahoo, eval_BoolQA, eval_QQP  # to be continued


# # add flash attention
# from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
# from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn
#
# replace_llama_attn_with_flash_attn()
# replace_bloom_attn_with_flash_attn()

def parse_args():
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
        "--inference_batch",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    #  add other inference params
    parser.add_argument(
        "--inference_tasks",
        type=str,
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
    
    parser.add_argument('--start_round',
                        default=0,
                        type=int,
                        help='which round (task) to start')
    
    parser.add_argument(
        "--lora_depth",
        type=int,
        default=-1,
        help="max depth of lora. -1 means no limit.",
    )
    
    parser.add_argument(
        "--gpus",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--master_port",
        type=int,
        required=True
    )
    # parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    gpus = args.gpus.split(',')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print('CUDA_VISIBLE_DEVICES:', os.environ["CUDA_VISIBLE_DEVICES"])
    total_rank = len(gpus)
    
    # set_random_seed(args.seed)
    # device = torch.device("cuda")
    
    def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                               ground_truths: list, round: int, i_task: int, task: str):
        # save as a json file
        df = {"eval"  : evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences,
              'labels': ground_truths}
        if not os.path.exists(args.inference_output_path):
            os.makedirs(args.inference_output_path)
        with open(args.inference_output_path + "/results-" + str(round) + "-" + str(i_task) + "-" + task + ".json", "w+", encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False)
    
    # set evaluation batch size
    # only support bs = 1, cause right padding training logic
    
    inference_tasks = (args.inference_tasks).split(',')
    task_num = len(inference_tasks)
    print("task_num: ", task_num, "inference_tasks: ", inference_tasks)
    # task_num: 8
    # inference_tasks: ['C-STANCE', 'FOMC', 'MeetingBank', 'Py150', 'ScienceQA', 'NumGLUE-cm', 'NumGLUE-ds', '20Minuten']
    start_round = copy.deepcopy(int(args.start_round))
    
    # del the start_round property in args:
    del args.start_round
    
    for round in range(start_round, task_num):  # load models and adapters of a new round in continual learning
        inference_model_path = os.path.join(args.inference_model_path, str(round))
        # print("Inference Model Path: " + inference_model_path, "local_rank" + args.local_rank)
        print("Inference Model Path: " + inference_model_path, "local_rank" + str(args.local_rank))
        
        # use command line of "deepspeed infer_part.py" to get the results:
        args.round = round
        args.total_rank = total_rank
        ranks = list(range(total_rank))
        # all_results_dic = run_inference(current_rank=, args)
        results = []
        # run one command:
        # run_inference(0, args)
        
        with ProcessPoolExecutor(max_workers=total_rank) as executor:
            futures = {executor.submit(run_inference, current_rank, args): current_rank for current_rank in ranks}
            
            for future in as_completed(futures):
                current_rank = futures[future]
                try:
                    result = future.result()
                    # print(f"RANK {current_rank} finished, result: {str(result)[:200]}")
                    # print blue:
                    print("\033[34m" + f"RANK {current_rank} finished, result: {str(result)[:200]}..." + "\033[0m")
                    
                    results.append((current_rank, result))
                except Exception as e:
                    print(f"RANK {current_rank} encountered an error: {e}")
        
        # for gpu_id, output in results:
        #     print(f"Output from RANK {gpu_id}: {output}")
        
        for inference_task_id in range(round + 1):  # evaluation for previous tasks in a single round
            inference_task = inference_tasks[inference_task_id]
            
            # sources_sequences, predicted_sequences, ground_truths = prediction(model, infer_dataloader)
            sources_sequences, predicted_sequences, ground_truths = [], [], []
            for rank, result in results:
                sources_sequences += eval(result[inference_task]['sources_sequences'])
                predicted_sequences += eval(result[inference_task]['predicted_sequences'])
                ground_truths += eval(result[inference_task]['ground_truths'])
            print(f'Task {inference_task_id} gathered, total len: {len(sources_sequences)}')
            
            base_dir = os.path.join('./check_output',
                                    f'{inference_task}_{args.CL_method}')
            os.makedirs(base_dir, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(base_dir, f'results_{timestamp}.json')

            # Prepare data to save
            save_data = {
                'sources_sequences': sources_sequences,
                'predicted_sequences': predicted_sequences,
                'ground_truths': ground_truths,
                'task_id': inference_task_id,
                'total_samples': len(sources_sequences)
            }

            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            print(f'Results saved to: {output_file}')

            # Get Accuracy/ROUGE/BLEU/...
            # The evaluation result is stored in a dictionary. e.g. {"accuracy": .., "rouge-L": ..}
            if inference_task == "ScienceQA":
                evaluation_result = eval_ScienceQA.eval(predicted_sequences, ground_truths)
            elif inference_task == "MeetingBank":
                evaluation_result = eval_MeetingBank.eval(predicted_sequences, ground_truths)
            elif inference_task == "C-STANCE":
                evaluation_result = eval_CStance.eval(predicted_sequences, ground_truths)
            elif inference_task == "Papyrus-f":
                evaluation_result = eval_PapyrusF.eval(predicted_sequences, ground_truths)
            elif inference_task == "Py150":
                evaluation_result = eval_Py150.eval(predicted_sequences, ground_truths)
            elif inference_task == "FOMC":
                evaluation_result = eval_FOMC.eval(predicted_sequences, ground_truths)
            elif inference_task == "NumGLUE-cm":
                evaluation_result = eval_NumGLUE_cm.eval(predicted_sequences, ground_truths)
            elif inference_task == "NumGLUE-ds":
                evaluation_result = eval_NumGLUE_ds.eval(predicted_sequences, ground_truths)
            elif inference_task == "20Minuten":
                evaluation_result = eval_20Minuten.eval(sources_sequences, predicted_sequences, ground_truths)
            elif inference_task == "amazon":
                evaluation_result = eval_amazon.eval(predicted_sequences, ground_truths)
            elif inference_task == "yelp":
                evaluation_result = eval_yelp.eval(predicted_sequences, ground_truths)
            elif inference_task == "agnews":
                evaluation_result = eval_agnews.eval(predicted_sequences, ground_truths)
            elif inference_task == "dbpedia":
                evaluation_result = eval_dbpedia.eval(predicted_sequences, ground_truths)
            elif inference_task == "yahoo":
                evaluation_result = eval_yahoo.eval(predicted_sequences, ground_truths)
            elif inference_task == "BoolQA":
                evaluation_result = eval_BoolQA.eval(predicted_sequences, ground_truths)
            elif inference_task == "QQP":
                evaluation_result = eval_QQP.eval(predicted_sequences, ground_truths)
            else:
                # default using accuracy
                evaluation_result = eval_QQP.eval(predicted_sequences, ground_truths)
            
            # if args.global_rank <= 0:  # only one process is running
            print("***** Saving inference results *****")
            save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, round, inference_task_id, inference_task)


def run_inference(current_rank, args):
    # random sleep 0-3s:
    time.sleep(current_rank * 2 + random.random())
    
    args.current_rank = current_rank

    gpu_list = args.gpus.strip().split(',')
    current_gpu = gpu_list[current_rank]
    
    dir_path = args.output_json_file_path = args.inference_model_path + '/intermediate_predictions'
    print("mkdir -p " + dir_path)
    os.system("mkdir -p " + dir_path)
    
    args.output_json_file_path = args.inference_model_path + '/intermediate_predictions/round{}_rank{}'.format(args.round, current_rank) + '.json'
    command = [
        "deepspeed",
        f"--include=localhost:{current_gpu}",
        f"--master_port {args.master_port + current_rank}",
        "inference/infer_part.py",
        "--deepspeed"
    ]
    # add other args:
    for key, value in vars(args).items():
        if key not in ["master_port", "gpus"]:
            command.append(f"--{key} {value}")
    
    # print red command:
    print("\033[31m" + " ".join(command) + "\033[0m")
    
    print('')
    # result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # os.system(" ".join(command))
    
    # os.popen(" ".join(command)).readlines()
    
    fail_num = 0
    result = None
    
    if fail_num < 3:
        try:
            # only need last line:
            _ = os.popen(" ".join(command)).readlines()[-2]
            # read str in args.output_json_file_path:
            with open(args.output_json_file_path, "r") as file:
                result = file.read()
        except Exception as e:
            fail_num += 1
            print(f"Fail for {fail_num} times. Error: {e}")
            result = None
            
    # # print red result:
    # print("\033[31m" + str(eval(result.strip())) + "\033[0m")
    
    # # result = subprocess.run(command, capture_output=True, text=True)
    # result = "".join(os.popen(" ".join(command)).readlines())
    return eval(result.strip())


if __name__ == "__main__":
    main()
