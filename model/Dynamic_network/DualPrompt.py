from copy import deepcopy
import torch
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
from model.base_model import CL_Base_Model
import numpy as np
from deepspeed.utils import safe_get_full_grad
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds
from transformers import GenerationConfig
import json
import os

generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)


def convert_DualPrompt_model(model, args):
    def init_new_prompt(prompt_len, embedding_dim):
        N = args.embed_tokens_length
        prompt_weights = []
        for i in range(prompt_len):
            with torch.no_grad():
                j = np.random.randint(N)  # random token
                w = deepcopy(args.embed_tokens.weight[j].detach().cpu().numpy())
                prompt_weights.append(w)
        return np.array(prompt_weights)
    
    # Initialize G-Prompt (shared across tasks)
    if args.prompt_init == 'uniform':
        model.model.g_prompt = nn.Parameter(
            torch.tensor(init_new_prompt(args.g_prompt_length, args.embed_tokens_dim),
                         requires_grad=True)
        )
    
    # Initialize E-Prompts (task-specific)
    model.model.e_prompts = nn.ParameterDict()
    model.model.task_keys = nn.ParameterDict()
    
    for task in args.train_task_list:
        model.model.e_prompts[task] = nn.Parameter(
            torch.tensor(init_new_prompt(args.e_prompt_length, args.embed_tokens_dim),
                         requires_grad=True)
        )
        # Initialize task-specific key
        model.model.task_keys[task] = nn.Parameter(
            torch.randn(args.embed_tokens_dim),
            requires_grad=True
        )
    
    return model


class DualPrompt(CL_Base_Model):
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        
        self.embed_dim = self.args.embed_tokens_dim
        self.embed_tokens = self.args.embed_tokens
    
    def fprompt(self, hidden_states, prompt):
        """Prompting function that combines hidden states with prompts"""
        return torch.cat([prompt, hidden_states], dim=1)
    
    def train_step(self, batch, task):
        input_ids = batch['input_ids']
        attn_masks = batch['attention_mask']
        labels = batch['labels']
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Apply G-Prompt (general prompt)
        g_prompted = self.fprompt(inputs_embeds, self.model.model.g_prompt.unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1))
        
        # Apply E-Prompt (task-specific prompt)
        e_prompt = self.model.model.e_prompts[task]
        e_prompted = self.fprompt(g_prompted, e_prompt.unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1))
        
        # Update attention mask for the additional prompt tokens
        total_prompt_length = self.args.g_prompt_length + self.args.e_prompt_length
        prompt_attention = torch.ones(input_ids.shape[0], total_prompt_length).to(attn_masks.device)
        attn_masks = torch.cat([prompt_attention, attn_masks], dim=1)
        
        # Extend labels for prompt tokens (typically ignored in loss computation)
        prompt_labels = torch.full((input_ids.shape[0], total_prompt_length), -100).to(labels.device)
        labels = torch.cat([prompt_labels, labels], dim=1)
        
        # Forward pass
        outputs = self.model(
            inputs_embeds=e_prompted,
            labels=labels,
            attention_mask=attn_masks,
            use_cache=False
        )
        
        loss = outputs[0]
        return loss
    
    def train_one_task(self, task, i_task, epochs):
        print('task = ', task)
        
        dataloader_train = self.train_task_list[task]
        self.train_length = len(dataloader_train)
        total_steps = epochs
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        
        for epoch in range(epochs):
            print(epoch)
            self.model.train()
            
            for step, batch in enumerate(tqdm(dataloader_train)):
                del batch['sources']
                batch = {k: batch[k].to('cuda') for k in batch}
                loss = self.train_step(batch, task)
                
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    description = f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                
                self.model.backward(loss)
                self.model.step()
    
    def evaluate_one_task(self, round, infer_task_id, task):
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
        
        infer_dataloader = self.test_task_list[task]
        progress_bar = tqdm(total=len(infer_dataloader), leave=True, disable=(self.args.global_rank != 0))
        
        def prediction(model, infer_dataloader):
            predicted_sequences = []
            sources_sequences = []
            label_sequences = []
            model.eval()
            
            for step, batch in enumerate(infer_dataloader):
                ground_truths_ids = self.tokenizer(
                    batch['gts'],
                    truncation=True,
                    max_length=self.args.max_ans_len,
                    add_special_tokens=False,
                    padding='max_length',
                    return_tensors='pt'
                )['input_ids'].to(device)
                
                del batch['gts']
                del batch['sources']
                batch = to_device(batch, device)
                progress_bar.update(1)
                
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    description = f"Step {step}"
                    progress_bar.set_description(description, refresh=False)
                
                with torch.no_grad():
                    input_ids = batch['input_ids']
                    attn_masks = batch['attention_mask']
                    inputs_embeds = self.embed_tokens(input_ids)
                    
                    # Apply prompts for inference
                    g_prompted = self.fprompt(inputs_embeds, self.model.model.g_prompt.unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1))
                    e_prompt = self.model.model.e_prompts[task]
                    e_prompted = self.fprompt(g_prompted, e_prompt.unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1))
                    
                    # Update attention mask
                    total_prompt_length = self.args.g_prompt_length + self.args.e_prompt_length
                    prompt_attention = torch.ones(input_ids.shape[0], total_prompt_length).to(attn_masks.device)
                    full_attn_masks = torch.cat([prompt_attention, attn_masks], dim=1)
                    
                    generate_ids = model.generate(
                        inputs_embeds=e_prompted,
                        attention_mask=full_attn_masks,
                        max_new_tokens=self.args.max_ans_len,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.unk_token_id,
                        generation_config=generation_config,
                        use_cache=False
                    )
                    
                    # Gather results for distributed training
                    gathered_ids, max_seq_len = self.dist_results_gather(generate_ids, self.tokenizer.eos_token_id)
                    gathered_labels, max_label_len = self.dist_results_gather(ground_truths_ids, self.tokenizer.eos_token_id)
                    
                    if self.args.global_rank <= 0:
                        sou_sequences = self.tokenizer.batch_decode(gathered_ids[:, : max_seq_len], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        pre_sequences = self.tokenizer.batch_decode(gathered_ids[:, max_seq_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        lab_sequences = self.tokenizer.batch_decode(gathered_labels[:, : max_label_len], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        predicted_sequences += pre_sequences
                        sources_sequences += sou_sequences
                        label_sequences += lab_sequences
            
            return sources_sequences, predicted_sequences, label_sequences
        
        # Inference
        print_rank_0("***** Start inference *****", self.args.global_rank)
        sources_sequences, predicted_sequences, ground_truths = prediction(self.model, infer_dataloader)
        
        # Evaluation
        if self.args.global_rank <= 0:
            evaluation_result = {}
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
            
            print("***** Saving inference results *****")
            self.save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, round, infer_task_id, task)
    
    def save_inference_results(self, evaluation_result, sources_sequences, predicted_sequences, ground_truths, round, i_task, task):
        df = {
            "eval"   : evaluation_result,
            'prompts': sources_sequences,
            'results': predicted_sequences,
            'labels' : ground_truths
        }
        
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        
        with open(f"{self.args.output_dir}/results-{round}-{i_task}-{task}.json", "w+", encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False)
    
    def evaluate(self, round, infer_task_id, task):
        self.evaluate_one_task(round, infer_task_id, task)
