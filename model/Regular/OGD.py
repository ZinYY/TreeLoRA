import json
import os

import torch
from torch.nn.utils.convert_parameters import _check_param_device, parameters_to_vector, vector_to_parameters
import torch.nn as nn
from tqdm.auto import tqdm
from deepspeed.utils import safe_get_full_grad
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0


def count_parameter(model):
    return sum(p.numel() for p in model.parameters())

def orthonormalize(vectors, normalize=True, start_idx=0):
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'
    # orthonormalized_vectors = torch.zeros_like(vectors)
    if normalize:
        vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)
    else:
        vectors[:, 0] = vectors[:, 0]

    if start_idx == 0 :
        start_idx = 1
    for i in tqdm(range(start_idx, vectors.size(1)), desc="orthonormalizing ..."):
        vector = vectors[:, i]
        V = vectors[:, :i]
        PV_vector = torch.mv(V, torch.mv(V.t(), vector))
        if normalize:
            vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)
        else:
            vectors[:, i] = (vector - PV_vector)

    return vectors

class OGD(CL_Base_Model):
    
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,max_memories=150):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        # if args.debug: #debug
        #     self.ogd_basis = torch.ones(n_params, 10, dtype=torch.bfloat16).to('cuda')
        # else:
        self.grad_dims = []
        for name, param in self.model.named_parameters():
            if "loranew_" in name:
                self.grad_dims.append(param.data.numel())
        self.cnt=len(self.grad_dims)
        self.max_memories = max_memories  #max memory for one task
        
        self.cur_grads = torch.zeros(sum(self.grad_dims), dtype=torch.bfloat16).cuda()
        n_params = sum(self.grad_dims)
        self.ogd_basis = torch.empty(n_params, 0, dtype=torch.bfloat16).to('cuda')

    
    def project_vec(self,gradient):
        raw_shape = gradient.shape
        beg = sum(self.grad_dims[:self.cnt-1])
        end = sum(self.grad_dims[:self.cnt])
        # print("begin:{}, end:{}".format(beg, end))
        self.cnt-=1
        proj_basis = self.ogd_basis[beg:end,:]
        
        if proj_basis.shape[1] > 0 and self.i_task!=0:  # param x basis_size
            # print("Using gradient project:{}".format(gradient.shape))
            gradient = gradient.view(-1)
            PV_vector = torch.mv(proj_basis, torch.mv(proj_basis.t(), gradient))
            gradient = (gradient - PV_vector)
            # if beg==0:
            #     print("raw grad:{}".format(gradient))
            #     print("projected_grads:{}".format(out.view(raw_shape)))
            return gradient.view(raw_shape)
        else:
            return gradient
        
    def store_grads(self, grads, grad_dims):
        # store the gradients
        # grads[:, tid].fill_(0.0)
        cnt = 0
        for name, param in self.model.named_parameters():
            if "loranew_" in name:
                hp_grad = safe_get_full_grad(param)
                if self.args.global_rank<=0:
                    if "embed" in name:
                        print(hp_grad)
                    if hp_grad is not None:
                        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                        en = sum(grad_dims[:cnt + 1])
                        grads[beg: en].copy_(torch.nan_to_num(hp_grad.data.view(-1),nan=0))
                    cnt += 1
    
        
    def train_step(self,
                    batch):

        lm_labels = batch["labels"]
        outputs = self.model(input_ids=batch['input_ids'], labels=lm_labels, attention_mask=batch['attention_mask'],use_cache=False)
        loss = outputs[0]

        return loss
    
    
    def train_one_task(self,
                       task,
                       i_task,
                       epochs,
                       ):

        # print('task = ', task)

        dataloader_train = self.train_task_list[task]

        self.new_basis=[]
        for epoch in range(epochs):
            # print(epoch)
            self.model.train()
            total_steps = epochs * len(dataloader_train)
            progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
            
            for step, batch in enumerate(tqdm(dataloader_train)):
                del batch['sources']
                batch = {k:batch[k].to('cuda') for k in batch}
                loss = self.train_step(batch)
                self.model.backward(loss)
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
        
                cnt=0
                for name, params in self.model.named_parameters():
                    if "loranew_" in name:
                        hp_grad = safe_get_full_grad(params)
                        beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                        en = sum(self.grad_dims[:cnt + 1])
                        self.cur_grads[beg: en].copy_(torch.nan_to_num(hp_grad.data.view(-1),nan=0))
                        cnt += 1
                        
                
                # if len(self.new_basis) < self.max_memories and epoch==0:
                if epoch==(epochs-1):
                    self.new_basis.append(self.cur_grads)
                        
                
                # cur_param = parameters_to_vector(self.get_params_dict())
                # grad_vec = self.parameters_to_grad_vector()
                # proj_grad_vec = project_vec(grad_vec, proj_basis=self.ogd_basis)
                
                # new_grad_vec = grad_vec - proj_grad_vec
                # lr = self.optimizer.state_dict()['base_optimizer_state']['param_groups'][0]['lr']
                # cur_param -= lr * new_grad_vec
                # vector_to_parameters(cur_param, self.get_params_dict())
                self.cnt = len(self.grad_dims)
                self.model.step()
                
                
            
        new_basis_tensor = torch.stack(self.new_basis).T
        self.ogd_basis = torch.cat([self.ogd_basis, new_basis_tensor], axis=1)
        self.ogd_basis = orthonormalize(self.ogd_basis, normalize=True)
        del new_basis_tensor
        del self.new_basis

    
    def train_continual(self):
        for name, params in self.model.named_parameters():
            print_rank_0('register_hook:' + name, self.args.global_rank)
            if "loranew_" in name:
                params.register_hook(lambda grads: self.project_vec(grads))
        for i_task, task in enumerate(self.train_task_list):
            self.i_task = i_task
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            self.save_model(i_task)

            # for j_task, _task in enumerate(self.train_task_list):
            #     if j_task > i_task:
            #         break
            #     self.evaluate(i_task, j_task, _task)
    
    def save_model(self, i_task):
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)
        
        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(i_task))
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
