import os
import time
import torch
import pickle
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, get_all_reduce_mean
from utils.data.data_utils import create_prompt_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def getInitialPrompt(tokenizer, prompt_token_number):
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-large")
    fr = open('model/Replay/allnumber.pickle', 'rb')
    all_tokens = pickle.load(fr)
    sorted_all_tokens = sorted(all_tokens.items(), key=lambda item: item[1], reverse=True)
    top5000_t5 = []
    for one in sorted_all_tokens:
        if len(top5000_t5) < 5000:
            top5000_t5.append(t5_tokenizer.decode(one[0]))

    top5000 = []
    vocab = []
    for token_id in range(len(tokenizer)):
        vocab.append(tokenizer.decode(token_id))
    for token in top5000_t5:
        if token in vocab:
            top5000.append(token)
        else: 
            random_token = random.choice(vocab)
            top5000.append(random_token)

    tokens_to_use = random.sample(top5000, prompt_token_number)
    initial_prompt = " ".join(tokens_to_use)
    input_ids = tokenizer.encode(initial_prompt)[:prompt_token_number]
    initial_prompt = tokenizer.decode(input_ids)

    return initial_prompt


class LFPT5(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        '''
        1. prompt tuning
        2. pseudo data generation
        '''
        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)


    def get_dataloader(self, task, pseudo_prompts, pseudo_answers):
        # get raw data
        dataset, _, _ = create_prompt_dataset(
            self.args.local_rank,
            os.path.join(self.args.data_path,task),
            self.args.data_output_path,
            self.args.seed
        )
        # add pseudo data
        if pseudo_prompts != None and pseudo_answers != None:
            dataset.prompt_dataset += pseudo_prompts
            dataset.answer_dataset += pseudo_answers
        # create dataLoaders
        if self.args.local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        from utils.data.data_collator import DataCollator
        collator  = DataCollator(
            self.tokenizer,
            padding="longest",
            max_prompt_len=self.args.max_prompt_len,
            max_ans_len=self.args.max_ans_len,
            pad_to_multiple_of=8,
            inference=False
        )
        dataloader = DataLoader(dataset,
                            collate_fn=collator,
                            sampler=sampler,
                            batch_size=self.args.per_device_train_batch_size)
        return dataloader


    def generate_pseudo_data(self, dataloader, task_name):
        print_rank_0(f"Generating pseudo data for " + task_name + "...", self.args.global_rank)
        pseudo_prompts = []
        pseudo_answers = []
        self.model.eval()
        for step, batch in enumerate(dataloader):
            pseudo_prompts += batch['sources']
            del batch['sources']
            batch = to_device(batch, self.device)
            prompt_len = batch['input_ids'].shape[1]
            with torch.no_grad():
                generate_ids = self.model.generate(input_ids=batch['input_ids'],
                                              attention_mask=batch['attention_mask'],
                                              max_new_tokens=self.args.max_ans_len,
                                              bos_token_id=self.tokenizer.bos_token_id,
                                              eos_token_id=self.tokenizer.eos_token_id,
                                              pad_token_id=self.tokenizer.unk_token_id,
                                              temperature=0.1,
                                              do_sample=True,
                                              num_return_sequences=1,
                                              use_cache=True
                                              )
            sequences = self.tokenizer.batch_decode(generate_ids[:, prompt_len:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
            pseudo_answers += sequences
        return pseudo_prompts, pseudo_answers


    def train_one_task(self, task, i_task, epochs):
        pseudo_prompts = []
        pseudo_answers = []
        for idx, previous_task in enumerate(self.train_task_list):
            if idx == i_task:
                break
            previous_dataloder = self.test_task_list[previous_task]
            prompts, answers = self.generate_pseudo_data(previous_dataloder, previous_task)
            pseudo_prompts += prompts
            pseudo_answers += answers
        train_dataloader = self.get_dataloader(task, pseudo_prompts, pseudo_answers)

        #### TRAIN ####
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()

        #### SAVE ####
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(i_task))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            self.model.save_pretrained(peft_model_id)  
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Successfully saving the final model to {peft_model_id}', self.args.global_rank)


    def save_model(self, i_task):
        pass
