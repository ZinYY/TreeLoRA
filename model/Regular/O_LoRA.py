import json
import os
import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from model.base_model import CL_Base_Model
from utils.model.model_utils import TIKTOK
from utils.utils import print_rank_0, to_device, get_all_reduce_mean


class O_LoRA(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 lamda_1 = 0.5, lamda_2 = 0
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


    def train_one_task(self, task, i_task, epochs):
        # if i_task > 0:
        #     self.lamda_2 = 0.1
        
        num_task = len(self.train_task_list)
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]

        #### TRAIN ####
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            self.tiktok = TIKTOK(self.args)
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()
            self.tiktok.print_time()
            
            # initialize:
            tmp_rounds = -1
            
            for step, batch in enumerate(train_dataloader):
                tmp_rounds += 1
                del batch['sources']
                batch = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                
                self.tiktok.tik()
                ########################### Regularization ##########################
                orthogonal_loss = 0.
                for name, param in self.model.named_parameters():
                    if "lora_A" in name:
                        for name_, param_ in self.model.named_parameters():
                            if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                                orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
                                break
                
                # l2-normalization for loranew_A/B
                l2_loss = 0.
                for name, param in self.model.named_parameters():
                    if "loranew_" in name:
                        l2_loss += torch.norm(param, p=2)


                loss = loss + orthogonal_loss * self.lamda_1 + l2_loss * self.lamda_2
                ######################################################################
                self.tiktok.tok('orthogonal loss time')
                
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                
                self.tiktok.tik()
                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()
                self.tiktok.tok('backward time')
                
                if self.args.global_rank == 0:
                    if tmp_rounds % 30 == 0:
                        print_rank_0(f"orthogonal_loss: {orthogonal_loss.item()}; l2_loss: {l2_loss.item()}; accuracy_loss: {loss.item()}; λ1: {self.lamda_1}; λ2: {self.lamda_2}", self.args.global_rank)
                        self.tiktok.print_time()
                
        
        # def split_string_by_first_num(s):
        #     for i, c in enumerate(s):
        #         if c.isdigit():
        #             return s[:i], s[i + 1:]
        #     return None, None
        #
        # #### COMBINE lora with lora_new and INITIALIZE lora_new ####
        # flag = 0
        # layer_id = 0
        # ## Different models may have different naming of modules.
        # ## 'setattr' is not work. So We have to hard code the name temporarily.
        # # layer_list = self.model.base_model.model.model.decoder.layers   # opt
        # layer_list = self.model.base_model.model.model.layers           # llama
        # state_dict = self.model.state_dict()
        # for k in state_dict:
        #     # # e.g. opt-1.3b
        #     # self.model.base_model.model.model.decoder.layers[layer_id].\
        #     # self_attn.v_proj.lora_A.default.weight.data\
        #     # = state_dict[k]
        #     if "v_proj.lora_A" in k:
        #         for k_ in state_dict:
        #             if ("v_proj.loranew_A" in k_) and (k.split("v_proj.lora_A")[0] == k_.split("v_proj.loranew_A")[0]):
        #                 state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=0) # [r_sum + r, dim]
        #                 layer_list[layer_id].self_attn.v_proj.lora_A.default.weight.data = state_dict[k]
        #                 break
        #         flag += 1
        #     elif "q_proj.lora_A" in k:
        #         for k_ in state_dict:
        #             if ("q_proj.loranew_A" in k_) and (k.split("q_proj.lora_A")[0] == k_.split("q_proj.loranew_A")[0]):
        #                 state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=0) # [r_sum + r, dim]
        #                 layer_list[layer_id].self_attn.q_proj.lora_A.default.weight.data = state_dict[k]
        #                 break
        #         flag += 1
        #     elif "v_proj.lora_B" in k:
        #         for k_ in state_dict:
        #             if ("v_proj.loranew_B" in k_) and (k.split("v_proj.lora_B")[0] == k_.split("v_proj.loranew_B")[0]):
        #                 state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=1) # [dim, r_sum + r]
        #                 layer_list[layer_id].self_attn.v_proj.lora_B.default.weight.data = state_dict[k]
        #                 break
        #         flag += 1
        #     elif "q_proj.lora_B" in k:
        #         for k_ in state_dict:
        #             if ("q_proj.loranew_B" in k_) and (k.split("q_proj.lora_B")[0] == k_.split("q_proj.loranew_B")[0]):
        #                 state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=1) # [dim, r_sum + r]
        #                 layer_list[layer_id].self_attn.q_proj.lora_B.default.weight.data = state_dict[k]
        #                 break
        #         flag += 1
        #     if flag == 4:
        #         layer_id += 1
        #         flag = 0
        # for k in state_dict:
        #     if "loranew_A" in k:
        #         nn.init.kaiming_uniform_(state_dict[k], a=math.sqrt(5))
        #     elif "loranew_B" in k:
        #         nn.init.zeros_(state_dict[k])
        # self.model.load_state_dict(state_dict)
        #
        # #### RESET ####
        # for name, param in self.model.named_parameters():
        #     if name.find("loranew_") != -1:
        #         param.requires_grad = True
        #     elif name.find("lora_") != -1:
        #         param.requires_grad = False

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
