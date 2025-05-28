from transformers import AutoTokenizer, LlamaForCausalLM
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from datasets import load_dataset
import torch
import torch.utils.data as Data
import random
from tqdm import tqdm
import json
import argparse
import math
from tqdm import tqdm
import pandas as pd
import json
import deepspeed
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


import torch.nn.functional as F
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)

from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model
from training.params import Method2Class, AllDatasetName
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten # to be continued


# os.environ['CUDA_VISIBLE_DEVICES']="0"


Constrained_PROMPT = "We will give you several examples and you should follow them to accomplish the task.\n Examples:\n"

# TASK_PROMT={
#     "FOMC":"What is the monetary policy stance for the following text? A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\n",
#     "C-STANCE":"判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n",
#     "ScienceQA":"Choose an answer for the following question and give your reasons.\n\n",
#     "NumGLUE-cm":"Solve the following math problem.\n",
#     "NumGLUE-ds":"Solve the following math problem.\n",
#     "MeetingBank":"Write a summary of the following meeting transcripts.\n",
#     "Py150":"Continue writing the code.\n",
#     "20Minuten":"Provide a simplified version of the following paragraph in German.\n\n",
#     "NIL":"What is the logical relationship between the \"sentence 1\" and the \"sentence 2\"? Choose one from the option.\n",
#     "QQP":"Whether the \"first sentence\" and the \"second sentence\" have the same meaning? Choose one from the option.\n",
#     "SC":"What is the sentiment of the following paragraph? Choose one from the option.\n",
#     "amazon":"What is the sentiment of the following paragraph? Choose one from the option.\nOption: very negative, negative, neutral, positive, very positive.\n",
#     "yelp": "What is the sentiment of the following paragraph? Choose one from the option.\nOption: very negative, negative, neutral, positive, very positive.\n",
#     "TC":"What is the topic of the following paragraph? Choose one from the option.\n",
#     "agnews": "What is the topic of the following paragraph? Choose one from the option.\nOption: World, Sports, Business, Science or Technology.\n",
#     "dbpedia": "What is the topic of the following paragraph? Choose one from the option.\nOption: Company, Educational Institution, Artist, Athlete, Office Holder, Mean of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film, Written Work.\n",
#     "yahoo": "What is the topic of the following paragraph? Choose one from the option.\nOption: Society & Culture, Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships, Politics & Government.\n",
#     "BoolQA":"According to the following passage, is the question true or false? Choose one from the option.\n",
#     "MultiRC":"According to the following passage and question, is the candidate answer true or false? Choose one from the option.\n",
#     "WiC":"Given a word and two sentences, whether the word is used with the same sense in both sentence? Choose one from the option.\n"
# }


TASK_PROMT = {
    "yelp"       : "What is the sentiment of the following paragraph? Choose one from the option.\nOption: very negative, negative, neutral, positive, very positive \n",
    "amazon"     : "What is the sentiment of the following paragraph? Choose one from the option.\nOption: very negative, negative, neutral, positive, very positive \n",
    "dbpedia"    : "What is the topic of the following paragraph? Choose one from the option.\nOption: Company, Educational Institution, Artist, Athlete, Office Holder, Mean of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film, Written Work \n",
    "yahoo"      : "What is the topic of the following paragraph? Choose one from the option.\nOption: Society & Culture, Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships, Politics & Government \n",
    "agnews"     : "What is the topic of the following paragraph? Choose one from the option.\nOption: World, Sports, Business, Science or Technology \n",
    "MNLI"       : "What is the logical relationship between the \"sentence 1\" and the \"sentence 2\"? Choose one from the option.\nOption: neutral, entailment, contradiction \n",
    "QQP"        : "Whether the \"first sentence\" and the \"second sentence\" have the same meaning? Choose one from the option.\nOption: False, True \n",
    "RTE"        : "What is the logical relationship between the \"sentence 1\" and the \"sentence 2\"? Choose one from the option.\nOption: contradiction, entailment \n",
    "SST-2"      : "What is the sentiment of the following paragraph? Choose one from the option.\nOption: Good, Bad \n",
    "WiC"        : "Given a word and two sentences, whether the word is used with the same sense in both sentence? Choose one from the option.\nOption: True, False \n",
    "CB"         : "What is the logical relationship between the \"sentence 1\" and the \"sentence 2\"? Choose one from the option.\nOption: entailment, contradiction, neutral \n",
    "COPA"       : "",
    "BoolQA"     : "According to the following passage, is the question true or false? Choose one from the option.\nOption: True, False \n",
    "MultiRC"    : "According to the following passage and question, is the candidate answer true or false? Choose one from the option.\nOption: False, True \n",
    "IMDB"       : "What is the sentiment of the following paragraph? Choose one from the option.\nOption: Good, Bad \n",
    "FOMC"       : "What is the monetary policy stance for the following text? A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\n",
    "NIL"        : "What is the logical relationship between the \"sentence 1\" and the \"sentence 2\"? Choose one from the option.\nOption: neutral, entailment, contradiction \n",
    "C-STANCE"   : "判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n",
    "ScienceQA"  : "Choose an answer for the following question and give your reasons.\n\n",
    "NumGLUE-cm" : "Solve the following math problem.\n",
    "NumGLUE-ds" : "Solve the following math problem.\n",
    "MeetingBank": "Write a summary of the following meeting transcripts.\n",
    "Py150"      : "Continue writing the code.\n",
    "20Minuten"  : "Provide a simplified version of the following paragraph in German.\n\n",
}


def collate_function(batch_prompt,demonstrations, task):
    processed_prompt = []
    for prompt in batch_prompt:

        task_prompt = TASK_PROMT[task]
        prompt = prompt[len(task_prompt):]
        demonstrations_prompt = ""
        for i in range(len(demonstrations["prompt"])):
            demonstrations_prompt += demonstrations["prompt"][i]
            demonstrations_prompt += demonstrations["answer"][i]
            demonstrations_prompt += "\n\n"
            
        prompt = task_prompt  +demonstrations_prompt + Constrained_PROMPT + prompt
        processed_prompt.append(prompt)
    return processed_prompt

class API:
    def __init__(self, model_path):
        self.path = model_path
        self.tokenizer=AutoTokenizer.from_pretrained(self.path)
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=LlamaForCausalLM.from_pretrained(self.path,device_map="auto")

    def api(self,prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
        # Generate
        outputs = self.model.generate(inputs.input_ids, max_length=2048, use_cache=True)
        # return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

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
    parser.add_argument('--dataset_name',
                    type=list_of_strings,
                    default='all',
                    help='Dataset to be used.')
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
    parser.add_argument('--demonstrations_num',
            default=int,
            help='Num of demonstrations')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args
def main():
    args = parse_args()
    # local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # Barrier to make sure all process are ready to train
    # torch.distributed.barrier()
    
    
    def dist_results_gather(generate_ids, pad_token=-1):
        # (batch_size, seq_len)
        result = generate_ids  # Example tensor
        local_batch_size = torch.tensor([result.size(0)], dtype=torch.int).cuda()
        local_seq_len = torch.tensor([result.size(1)], dtype=torch.int).cuda()

        global_batch_sizes = [torch.tensor([0], dtype=torch.int).cuda() for _ in range(dist.get_world_size())]
        global_seq_len = [torch.tensor([0], dtype=torch.int).cuda() for _ in range(dist.get_world_size())]
        dist.all_gather(global_batch_sizes, local_batch_size)
        dist.all_gather(global_seq_len, local_seq_len)

        max_seq_len = max([int(seq_len.item()) for seq_len in global_seq_len])

        if result.size(1) < max_seq_len:
            pad_seq_len = (max_seq_len - result.size(1), 0)
            result = F.pad(result, pad_seq_len, "constant", pad_token).cuda()

        total_results = [torch.zeros((int(bs.item()), max_seq_len), dtype=result.dtype).cuda() for bs in global_batch_sizes]
        dist.all_gather(total_results, result)

        flat_results = torch.cat(total_results, dim=0)

        return flat_results, max_seq_len
    
    def prediction(model, infer_dataloader, task):
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        model.eval()

        for step, batch in enumerate(infer_dataloader):
            # sources_sequences += batch['sources']
            ground_truths += batch['gts']
            ground_truths_ids = tokenizer(batch['gts'], 
                                            truncation=True,
                                            max_length=args.max_ans_len,
                                            add_special_tokens=False,
                                            padding='max_length',
                                            return_tensors='pt')['input_ids'].to(device)
            del batch['sources']
            del batch['gts']
            batch = to_device(batch, device)

            # update progress bar
            if args.global_rank == 0:
                progress_bar.update(1)
                description = f"Step {step}"
                progress_bar.set_description(description, refresh=False)

            with torch.no_grad():
                # sft config
                generate_ids = model.generate(input_ids=batch['input_ids'],
                                            attention_mask=batch['attention_mask'],
                                            max_new_tokens=args.max_ans_len,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.unk_token_id,
                                            generation_config=generation_config,
                                            use_cache=True
                                            )
                
            # add for distributed 
            # gathered_ids, max_seq_len = dist_results_gather(generate_ids, tokenizer.eos_token_id)
            # gathered_labels, max_label_len = dist_results_gather(ground_truths, tokenizer.eos_token_id)
            max_seq_len = batch['input_ids'].shape[1]

            if args.global_rank <= 0:
                sou_sequences = tokenizer.batch_decode(generate_ids[:, : max_seq_len], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if task=="FOMC" or task=="C-STANCE":
                    pre_sequences = tokenizer.batch_decode(generate_ids[:, max_seq_len:max_seq_len+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                else:
                    pre_sequences = tokenizer.batch_decode(generate_ids[:, max_seq_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

                if "NumGLUE" in task:
                    for i in range(len(pre_sequences)):
                        pre_sequences[i] = pre_sequences[i].split("\n")[0]
                elif "MeetingBank" in task:
                    for i in range(len(pre_sequences)):
                        pre_sequences[i] = pre_sequences[i].split("Meeting transcripts")[0]
                elif "ScienceQA" in task:
                    for i in range(len(pre_sequences)):
                        pre_sequences[i] = pre_sequences[i].split("Question:")[0]
                elif "Py150" in task:
                    for i in range(len(pre_sequences)):
                        pre_sequences[i] = pre_sequences[i].split("<EOL>")[0]
                elif "20Minuten" in task:
                    for i in range(len(pre_sequences)):
                        pre_sequences[i] = pre_sequences[i].split("Paragraph")[0]
                predicted_sequences += pre_sequences
                sources_sequences += sou_sequences

        return sources_sequences, predicted_sequences, ground_truths
    
    def get_random_demonstrations(dem_num, infer_dataset, length_limit, task):
        length_limit_per_sample = length_limit/(dem_num*2)
        demonstrations=[]
        answers = []
        i=0
        round=0
        while i<dem_num:
            round+=1
            if round==10000:
                break
            demonstration_id = random.randint(0,len(infer_dataset)-1)
            demonstration=infer_dataset[demonstration_id]  #[{prompt*4},{answer*4}]
            if task!="Py150":
                demonstration["prompt"] = demonstration["prompt"][len(TASK_PROMT[task]):]
            if len(tokenizer(demonstration["prompt"])['input_ids'])+ len(tokenizer(demonstration["answer"])['input_ids']) <= length_limit_per_sample:
                if task=="FOMC" or task=="C-STANCE":
                    if answers.count(demonstration["answer"])<dem_num/3:
                        demonstrations.append(demonstration)
                        answers.append(demonstration["answer"])
                        i+=1
                else:
                    if demonstration["answer"] not in answers:
                        demonstrations.append(demonstration)
                        answers.append(demonstration["answer"])
                        i+=1
            else:
                continue
            
            if len(demonstrations)==dem_num:
                return demonstrations

        return demonstrations
        
        


    # set evaluation batch size
    # only support bs = 1, cause right padding training logic
    from transformers import GenerationConfig
    generation_config = GenerationConfig(
        temperature=args.temperature,
        do_sample=True,
        num_return_sequences=1
    )
    if args.dataset_name[0] == "all":
        Datasets = AllDatasetName
    else:
        Datasets = args.dataset_name    
    
    model_path = args.model_name_or_path

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"
    model = create_hf_model(AutoModelForCausalLM,
                                args.model_name_or_path,
                                tokenizer,
                                ds_config=None,
                                )
    replace_with_kernel_inject = False if "falcon" in args.model_name_or_path.lower() else True
    ds_engine = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.bfloat16, checkpoint=None,
                                            replace_with_kernel_inject=replace_with_kernel_inject,
                                            max_out_tokens=args.max_prompt_len + args.max_ans_len)
    model = ds_engine.module


    for task in Datasets:
        data_path = args.data_path
        inference_output_path = args.inference_output_path
        inference_output_path = os.path.join(inference_output_path, task)
        
        dataset_path = os.path.join(data_path, task)
        
        _, _, infer_dataset = create_prompt_dataset(
                args.local_rank,
                dataset_path,
                args.data_output_path,
                args.seed
            )
        
        demonstrations = get_random_demonstrations(int(args.demonstrations_num), infer_dataset, args.max_prompt_len-len(tokenizer(TASK_PROMT[task]+Constrained_PROMPT)['input_ids']),task)
        print_rank_0("demonstrations length:{}".format(len(demonstrations)),args.global_rank)
        if task=="MeetingBank":
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
                task=task
            )

        infer_sampler = SequentialSampler(infer_dataset)
        infer_dataloader = DataLoader(infer_dataset,
                                        collate_fn=inf_data_collator,
                                        sampler=infer_sampler,
                                        batch_size=args.inference_batch)

        
        progress_bar = tqdm(total=len(infer_dataloader), leave=True)
        print_rank_0("***** Start inference *****", args.global_rank)
        sources_sequences, predicted_sequences, ground_truths = prediction(model, infer_dataloader, task)
        # for step, batch in enumerate(test_loader):
        #     batch_prompt = batch["prompt"]
        #     answer = batch["answer"]
            
        #     #demonstrations
        #     demonstrations_id = [random.randint(0,len(test_dataset)-1) for i in range(demonstrations_num)]
        #     demonstrations=test_dataset[demonstrations_id]  #[{prompt*4},{answer*4}]
        #     demonstrations["prompt"] = [prompt[len(TASK_PROMT[task]):] for prompt in demonstrations["prompt"]]
        #     batch_prompt = collate_function(batch_prompt, demonstrations, task)
            
        #     if task=="FOMC" or task=="C-STANCE":
        #         output = [output[i][len(batch_prompt[i]):].split("\n")[0] for i in range(len(output))]
                
        #         ground_truths += answer
        #         predicted_sequences += output
        #         sources_sequences += batch_prompt
        #         # break
        #         progress_bar.update(1)
        #         description = f"Step {step}"
        #         progress_bar.set_description(description, refresh=False)
            
        
        if task == "ScienceQA":
            evaluation_result = eval_ScienceQA.eval(predicted_sequences, ground_truths)
        elif task == "MeetingBank":
            evaluation_result = eval_MeetingBank.eval(predicted_sequences, ground_truths)
        elif task == "C-STANCE":
            evaluation_result = eval_CStance.eval(predicted_sequences, ground_truths)
        elif task == "Papyrus-f":
            evaluation_result = eval_PapyrusF.eval(predicted_sequences, ground_truths)
        elif task == "Py150":
            evaluation_result = eval_Py150.eval(predicted_sequences, ground_truths)
        elif task == "FOMC":
            evaluation_result = eval_FOMC.eval(predicted_sequences, ground_truths)
        elif task == "NumGLUE-cm":
            evaluation_result = eval_NumGLUE_cm.eval(predicted_sequences, ground_truths)
        elif task == "NumGLUE-ds":
            evaluation_result = eval_NumGLUE_ds.eval(predicted_sequences, ground_truths)
        elif task == "20Minuten":
            evaluation_result = eval_20Minuten.eval(sources_sequences, predicted_sequences, ground_truths)
            
            
            
        print(evaluation_result)
        df = {"eval": evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences,
            'labels': ground_truths}
        
        if not os.path.exists(inference_output_path):
            os.makedirs(inference_output_path)
            
        with open(inference_output_path + "/results-" + task + ".json", "w+", encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False)


if __name__ == "__main__":
    main()