import numpy as np
import torch
import quadprog
import random
from tqdm.auto import tqdm
from deepspeed.utils import safe_get_full_grad
from model.base_model import CL_Base_Model
from qpth.qp import QPFunction
from utils.utils import print_rank_0


class GEM(CL_Base_Model):
    def __init__(self,model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.observed_tasks = []
        # for name, param in self.model.named_parameters():
        #     self.grad_dims.append(param.data.numel())
        self.n_tasks = len(self.train_task_list.keys())
        self.grads = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad==True:
                self.grads[name] = torch.zeros([param.data.numel(), self.n_tasks], dtype=torch.bfloat16)

                
    def store_grads(self, grads, tid):
        """
            This stores parameter gradients of past tasks.
            pp: parameters
            grads: gradients
            grad_dims: list with number of parameters per layers
            tid: task id
        """
        # store the gradients
        # grads[:, tid].fill_(0.0)
        print_rank_0("Storing gradient................", self.args.global_rank)
        for name, param in self.model.named_parameters():
            if param.requires_grad==True:
                hp_grad = safe_get_full_grad(param)
                if hp_grad is not None:

                    grads[name][:,tid].copy_(hp_grad.data.view(-1).cpu())
                
    def project2cone2(self, name, gradient, indx, margin=0.5, eps=1e-3):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            
        """
        # print_rank_0(gradient.shape,self.args.global_rank)
        # print("rank {}:{}".format(self.args.global_rank, gradient.shape))
        raw_shape = gradient.shape
        gradient = gradient.view(-1)
        # beg = sum(self.grad_dims[:self.cnt-1])
        # end = sum(self.grad_dims[:self.cnt])
        # self.cnt-=1
        # # print(self.cnt)

        # memories = self.grads[beg:end].cuda().index_select(1,indx)
        
        memories = self.grads[name].cuda().index_select(1,indx)
        # dotp = torch.mm(self.grads[:, i_task].unsqueeze(0),
        #                         self.grads.index_select(1, indx))
        dotp = torch.mm(gradient.unsqueeze(0), memories)
        
        gradient = gradient.unsqueeze(1)

        
        if (dotp < 0).sum() != 0:
        #     # memories_np = memories.cpu().t().double().numpy()
        #     # gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        #     # t = memories_np.shape[0]
        #     # P = np.dot(memories_np, memories_np.transpose())
        #     # P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        #     # q = np.dot(memories_np, gradient_np) * -1
        #     # G = np.eye(t)
        #     # h = np.zeros(t) + margin
        #     # v = quadprog.solve_qp(P, q, G, h)[0]
        #     # x = np.dot(v, memories_np) + gradient_np
        #     # gradient.copy_(torch.Tensor(x).view(-1, 1))
            memories_cuda = memories.t()
            gradient_cuda = gradient.contiguous().view(-1)
            t = memories_cuda.shape[0]
            # print("memories_cuda.shape:", memories_cuda.shape, " gradient_cuda.shape", gradient_cuda.shape, " t:", t)
            P = torch.matmul(memories_cuda, memories_cuda.t())
            P = 0.5 * (P + P.t()) + torch.eye(t).cuda() * eps
            q = torch.matmul(memories_cuda, gradient_cuda) * -1
            
            P = P.to(torch.float32)
            q = q.to(torch.float32)
            G = torch.eye(t).cuda()
            h = torch.zeros(t).cuda() + margin
            e = torch.Tensor().cuda()
            # print("P.type():", P.type(), " q.type():", q.type(), " G.type():", G.type(), " h.type():", h.type)
            v = QPFunction(verbose=False)(P, q, G, h, e, e)[0]
            v = v.to(torch.bfloat16)
            x = torch.matmul(v, memories_cuda) + gradient_cuda
            gradient.copy_(x.view(-1, 1))
            
        return gradient.view(raw_shape)
                    
    def train_step(self,
                    batch):

        lm_labels = batch["labels"]
        outputs = self.model(input_ids=batch['input_ids'], labels=lm_labels, attention_mask=batch['attention_mask'],use_cache=False)
        loss = outputs[0]

        return loss
    
    
    def train_one_task(self,
                       task,
                       i_task,
                       epochs=40
                       ):
        

        self.observed_tasks.append(i_task)

        dataloader_train = self.train_task_list[task]
        indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) 
        # if indx.shape[0]!=0:
        # for name, params in self.model.named_parameters():
            
        #     params.register_hook(lambda grads: self.project2cone2(grads,indx))
        for epoch in range(epochs):
            self.model.train()
            total_steps = epochs * len(dataloader_train)
            progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

            for step, batch in enumerate(tqdm(dataloader_train)):
                del batch['sources']
                batch = {k:batch[k].to('cuda') for k in batch}
                loss = self.train_step(batch)
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                
                self.model.backward(loss)
                if i_task!=0:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad!=None:
                            param.grad.data.copy_(self.project2cone2(name, param.grad, indx))
                # for name, param in self.model.named_parameters():
                #     hp_grad = safe_get_full_grad(param)
                #     if self.args.global_rank<=0:
                #         print("{}:{}".format(name,hp_grad))
                
                if step == len(dataloader_train)-1:
                    self.store_grads(
                            grads=self.grads,
                            tid=i_task)
                    
                self.model.step()



            
