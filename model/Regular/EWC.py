from copy import deepcopy

import torch
from torch.autograd import Variable
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
from model.base_model import CL_Base_Model
from utils.model.model_utils import TIKTOK


class EWC(CL_Base_Model):
    def __init__(self,model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args, lambda_ewc=400):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.device="cuda"
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._previous_params = {}

        for n, p in deepcopy(self.params).items():
            self._previous_params[n] = p.data.cpu() # Previous task parameters
        self.grads = {}

        self.tiktok = TIKTOK(args)
        
        self.fisher = {}
        self.init_fisher()
        del self.params


    
    def init_fisher(self):
        for n, p in deepcopy(self.params).items():
            if p.requires_grad==True:
                p.data.zero_()
                self.fisher[n] = p.data
            
    def _update_fisher(self):
        for n, p in self.model.named_parameters():
            if n in self.grads.keys():
                self.fisher[n].data += self.grads[n].cuda().data ** 2 / self.train_length
                
    def _regular_fisher(self):
        for n, p in self.model.named_parameters():
            if n in self.grads.keys():
                self.fisher[n].data /= self.train_length

    
    def _update_previous_params(self):
        for n, p in self.model.named_parameters():
            self._previous_params[n] = p.data.cpu() # Previous task parameters


    def penalty(self):
        restrict_loss = 0
        precision_matrices = self.fisher
        for n, p in self.model.named_parameters():
            if p.requires_grad==True:
                restrict_loss_params = precision_matrices[n] * (p - self._previous_params[n].cuda()) ** 2
                restrict_loss += restrict_loss_params.sum()
        return restrict_loss
    
    def train_step(self,
                    batch):

        # batch = {k: batch[k].to(self.device) for k in batch}
        lm_labels = batch["labels"]
        # lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        # inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
        # inputs_embeds = self.model.model.embed_tokens(batch["input_ids"])  #[batch * embedding_size]
        
        self.tiktok.tik()
        outputs = self.model(input_ids=batch['input_ids'], labels=lm_labels, attention_mask=batch['attention_mask'],use_cache=False)
        self.tiktok.tok('forward_time')
        
        loss = outputs[0]
        
        # ## add L1-norm here:
        # # l1_norm = torch.norm(self.model.model.encoder.embed_tokens.weight, p=1)
        # l1_norm = sum(p.abs().sum() for p in self.model.parameters())
        # loss += 50.0 * l1_norm
        
        if self.task_num!=0:
            restrict_loss = self.penalty()
            loss += 0.5*self.lambda_ewc*restrict_loss

        return loss
    
    def save_grad(self,name):
        def hook(grad):
            grad = torch.nan_to_num(grad, nan=0)
            # grad = torch.clamp(grad, -self.args.ds_config['gradient_clipping'], self.args.ds_config['gradient_clipping'])
            self.grads[name] = grad.cpu()
            del grad
        return hook
    def retain_grad(self):
        for n,p in self.model.named_parameters():
            if n in self.fisher.keys():
                p.register_hook(self.save_grad(n))
    
    
    def train_one_task(self,
                       task,
                       i_task,
                       epochs=40):

        print('task = ', task)

        dataloader_train = self.train_task_list[task]
        self.train_length = len(dataloader_train)
        total_steps = epochs * len(dataloader_train)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))


        for epoch in range(epochs):
            print(epoch)
            self.model.train()

            for step, batch in enumerate(tqdm(dataloader_train)):
                del batch['sources']
                batch = {k:batch[k].to('cuda') for k in batch}
                loss = self.train_step(batch)
                
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                self.tiktok.tik()
                self.model.backward(loss)
                self.model.step()
                self.tiktok.tok('backward time')
                
                self.tiktok.tik()
                self._update_fisher()
                self.tiktok.tok( 'update fisher time')


    
    # Train model continually
    def train_continual(self):
        self.retain_grad()

        for i_task, task in enumerate(self.train_task_list):
            self.task_num=i_task
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            # self._regular_fisher()
            
            self._update_previous_params()
            self.save_model(i_task)
            
            


