import copy
import json
import os
import pickle
import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from model.base_model import CL_Base_Model
from utils.kd_lora_tree import KD_LoRA_Tree
from utils.model.model_utils import TIKTOK
from utils.utils import print_rank_0, to_device, get_all_reduce_mean


class Tree_LoRA(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 lamda_1=0.5, lamda_2=0
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        '''
        orthological to previous adapters
        '''
        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        self.tiktok = TIKTOK(args)
        
        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)
        
        num_task = len(self.train_task_list)
        args.num_tasks = num_task
        self.kd_lora_tree = KD_LoRA_Tree(args)
    
    def train_one_task(self, task, task_id, epochs):
        # if task_id > 0:
        #     self.lamda_2 = 0.1
        
        num_task = len(self.train_task_list)
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        
        #### TRAIN ####
        total_steps = epochs * len(train_dataloader)
        train_dataloader_len = len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        
        for epoch in range(epochs):
            self.tiktok = TIKTOK(self.args)
            print_rank_0(
                f"Beginning of Epoch {epoch + 1}/{epochs}, Total Micro Batches {train_dataloader_len}",
                self.args.global_rank)
            self.model.train()
            self.tiktok.print_time(self.args.global_rank)
            
            self.kd_lora_tree.new_epoch_init(train_dataloader_len)
            tmp_rounds = -1
            
            for step, batch in enumerate(train_dataloader):
                tmp_rounds += 1
                
                if self.args.reg > 0:
                    self.kd_lora_tree.step()
                    
                del batch['sources']
                batch = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                
                
                if self.args.reg > 0:
                    self.tiktok.tik()
                    # get _grad_current:
                    _grad_current = []
                    for name_, param_ in self.model.named_parameters():
                        if "loranew_A" in name_:
                            _grad_current.append(param_)  # add [r * dim]
                    self.tiktok.tok("Calculate_Grad_@Task{} Epoch{}".format(task_id, epoch))
                    
                    self.tiktok.tik()
                    # change dimension
                    _shape = _grad_current[0].shape
                    _grad_current = torch.stack([_grad_current[i].reshape(-1) for i in range(len(_grad_current))], dim=0)
                    # _grad_current: (lora_depth, dim * rank * para_nums)
                    
                    self.kd_lora_tree.insert_grad(_grad_current)
                    
                    self.tiktok.tok("Split_Grad_@Task{} Epoch{}".format(task_id, epoch))
                    
                    if task_id > 0:
                        self.tiktok.tik()
                        prev_id_matrix = self.kd_lora_tree.tree_search(task_id, device=self.device)
                        self.tiktok.tok("Calculate_Tree_Search_@Task{} Epoch{}".format(task_id, epoch))
                        
                        self.tiktok.tik()
                        # vectorize to accelerate this part:
                        reg_loss = self.kd_lora_tree.get_loss(_grad_current, loss, task_id, prev_id_matrix)
                        # reg_loss = agem_regularizer(_grad_current, all_grad_device, task_id, prev_id_matrix, target, device, args)
                        
                        # reg_loss = reg_loss / (reg_loss.detach().clone().item() + 1e-5) * (loss.detach().clone().item())
                        
                        loss = loss - reg_loss
                        self.tiktok.tok("Calculate_Tree_Reg_@Task{} Epoch{}".format(task_id, epoch))
                        
                        if tmp_rounds % 100 == 0:
                            print_rank_0("\033[34m(Normal Process) Sim: {};\033[0m".format(self.kd_lora_tree.sim), self.args.global_rank)
                            print_rank_0("\033[34m(Normal Process) Selected Nums: {};\033[0m".format(self.kd_lora_tree.num_of_selected[:task_id]), self.args.global_rank)
                            print_rank_0("\033[34m(Normal Process) Prev_id_matrix: {};\033[0m".format(prev_id_matrix), self.args.global_rank)
                            print_rank_0("\033[34mReg Loss: {:.4f}\033[0m".format(reg_loss), self.args.global_rank)
                            # print blue:
                        
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                
                self.tiktok.tik()
                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()
                self.tiktok.tok('backward time')
                
                if self.args.global_rank == 0:
                    if tmp_rounds % 30 == 0:
                        self.tiktok.print_time()
                
        
        #### SAVE ####
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)
        
        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(task_id))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Successfully saving the final model to {peft_model_id}', self.args.global_rank)
            
            # if self.args.reg > 0:
            #     # save the tree using pickle:
            #     with open(os.path.join(peft_model_id, 'treelora_task_{}.pkl'.format(task_id)), 'wb') as f:
            #         pickle.dump(self.kd_lora_tree, f)
        
        if self.args.reg > 0:
            # after each task:
            self.kd_lora_tree.end_task(task_id=task_id)
    
    # def save_model(self, i_task):
    #     pass
    
    def save_model(self, round):
        # # if self.args.output_dir is not None:
        # #     print_rank_0('saving the final model ...', self.args.global_rank)
        # #
        # # if self.args.global_rank == 0:
        # #     peft_model_id = os.path.join(self.args.output_dir, str(i_task))
        # #     if not os.path.exists(peft_model_id):
        # #         os.makedirs(peft_model_id)
        # #     self.model.save_pretrained(peft_model_id)
        # #     self.tokenizer.save_pretrained(peft_model_id)
        # #     print_rank_0(f'Successfully saving the final model to {peft_model_id}', self.args.global_rank)
        #
        # #### RESET ####
        # for name, param in self.model.named_parameters():
        #     if name.find("loranew_") != -1:
        #         param.requires_grad = True
        #     elif name.find("lora_") != -1:
        #         param.requires_grad = False
        #
        #### SAVE ####
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)
        
        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(round))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            adapter_config_path = os.path.join(peft_model_id, 'adapter_config.json')
            # read json, load to adapter_config:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            # change "r_sum" in adapter_config to 0:
            adapter_config['r_sum'] = 0  # This is the key point to be compatible with O_LoRA!!!
            # save to json:
            with open(adapter_config_path, 'w') as f:
                json.dump(adapter_config, f)
            print_rank_0(f'Successfully saving the final model to {peft_model_id}', self.args.global_rank)

