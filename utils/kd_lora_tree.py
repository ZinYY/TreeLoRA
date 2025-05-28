import copy

import math

import torch

from utils.utils import print_rank_0


class KDTreeNode:
    def __init__(self, task_indices, depth, grads_tensor, lora_depth):
        """
        :param depth: current depth
        :param grads_tensor: (num_tasks, lora_depth, feature_dim)
        """
        self.task_indices = task_indices  # List of global task indices in this node
        self.depth = depth  # Current depth of the node
        self.left = None  # Left child
        self.right = None  # Right child
        self.is_leaf = False  # Flag to check if it's a leaf node
        self.lora_depth = lora_depth  # Maximum depth of the tree
        self.mean_vector = None  # Mean vector at this node
        self.median_similarity = None  # Median similarity for splitting
        
        self.build_node(grads_tensor)
    
    def build_node(self, grads_tensor):
        if self.depth >= self.lora_depth or len(self.task_indices) <= 1:
            self.is_leaf = True
            return
        
        current_grads = grads_tensor[self.task_indices, self.depth, :]  # Shape: (N, D)
        
        self.mean_vector = current_grads.mean(dim=0)  # Shape: (D,)
        
        similarities = torch.mv(current_grads, self.mean_vector)  # Shape: (N,)
        
        self.median_similarity = torch.median(similarities).item()
        
        left_indices = [
            self.task_indices[i]
            for i in range(len(self.task_indices))
            if similarities[i].item() >= self.median_similarity
        ]
        right_indices = [
            self.task_indices[i]
            for i in range(len(self.task_indices))
            if similarities[i].item() < self.median_similarity
        ]
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            median = len(self.task_indices) // 2
            left_indices = self.task_indices[:median]
            right_indices = self.task_indices[median:]
        
        self.left = KDTreeNode(left_indices, self.depth + 1, grads_tensor, self.lora_depth)
        self.right = KDTreeNode(right_indices, self.depth + 1, grads_tensor, self.lora_depth)
    
    def __str__(self, level=0):
        indent = "  " * level
        if self.is_leaf:
            return f"{indent}Leaf(depth={self.depth}, tasks={self.task_indices})\n"
        else:
            mean_list = self.mean_vector[:2].tolist()
            mean_str = ", ".join([f"{x:.4f}" for x in mean_list])
            result = (
                f"{indent}Node(depth={self.depth}, tasks={self.task_indices}, "
                f"mean_vector=[{mean_str}, ...], median_similarity={self.median_similarity:.4f})\n"
            )
            if self.left:
                result += self.left.__str__(level + 1)
            if self.right:
                result += self.right.__str__(level + 1)
            return result

def tree_lora_loss(current_grad, all_grad, task_id, prev_id_matrix, multiple_module=True):
    # calculate the gradient of the current task's similarity
    reg_loss = None
    if multiple_module:
        for (depth_id, prev_task_id) in enumerate(prev_id_matrix):
            if reg_loss is None:
                reg_loss = -(current_grad[depth_id] * all_grad[prev_task_id][depth_id]).sum()
            else:
                reg_loss += -(current_grad[depth_id] * all_grad[prev_task_id][depth_id]).sum()
        
    else:
        prev_id = prev_id_matrix[0]
        reg_loss = (-(current_grad.reshape(-1) * all_grad[prev_id].reshape(-1)).sum())
        
    return reg_loss


class KD_LoRA_Tree:
    def __init__(self, args):
        """
        Initialize the KD-tree.

        Args:
            points (Tensor): A tensor of shape (N, D) where N is the number of points and D is the dimension.
            leaf_size (int): The maximum number of points in a leaf node before rebuilding.
        """
        self.root = None
        
        self.mask = None
        self.mask_tensor = None
        self.last_task_id = -1
        self.args = args
        self.all_grad_device = None
        self.all_accumulate_grads = [None] * self.args.num_tasks
        
        self.num_of_selected = None
        
        self.kd_tree_root = None  # Initialize the KD Tree root
        self.current_grad = None  # Placeholder for the current gradient
        self.sim = None  # Initialize similarity tensor
    
    def new_epoch_init(self, train_dataloader_len):
        # initialize:
        self.current_grad = None
        self.all_grad = None
        self.num_of_selected = None
        self.tmp_rounds = -1
        self.total_rounds = train_dataloader_len
        self.sim = None  # Reset similarity tensor at the start of each epoch
    
    def end_task(self, task_id):
        # after each task:
        if self.args.reg > 0:
            self.all_accumulate_grads[task_id] = self.current_grad
        
        lora_depth = self.current_grad.shape[0]
        
        # update the tree according to the all_accumulate_grads
        
        print(f"Updating the KD Tree with task {task_id}...")
        print_rank_0(f"\nUpdating the KD Tree with task {task_id}...", self.args.global_rank)
        
        valid_grads = self.all_accumulate_grads[:task_id + 1]
        valid_grads = [grad for grad in valid_grads if grad is not None]
        
        if not valid_grads:
            print("No gradients to build the tree.")
            return
        
        # (num_valid_tasks, lora_depth, feature_dim)
        grads_tensor = copy.deepcopy(torch.stack(valid_grads))
        
        for i in range(grads_tensor.shape[0] - 1, 0, -1):
            grads_tensor[i] = grads_tensor[i] - grads_tensor[i - 1]  # calculate difference
            
        num_valid_tasks = grads_tensor.shape[0]
        
        task_ids = [
            i for i, grad in enumerate(self.all_accumulate_grads[:task_id + 1])
            if grad is not None
        ]
        
        self.kd_tree_root = KDTreeNode(
            task_indices=task_ids,
            depth=0,
            grads_tensor=grads_tensor,
            lora_depth=lora_depth
        )
        
        print_rank_0("KD Tree updated successfully.", self.args.global_rank)
        # print("KD Tree Structure:")
        print_rank_0(self.kd_tree_root, self.args.global_rank)
    
    def step(self):
        self.tmp_rounds += 1
        self.tmp_reg = self.args.reg * self.tmp_rounds / self.total_rounds
    
    def insert_grad(self, _grad_current):
        for i in range(len(_grad_current)):
            if self.current_grad is None:
                # current_grad = _grad_current.detach().to('cpu', non_blocking=True) * 1.0 / total_rounds
                self.current_grad = _grad_current.detach() * 1.0 / self.total_rounds
            else:
                frac = 1.0 / self.total_rounds
                self.current_grad += _grad_current.detach() * frac
        # current_grad: (lora_depth, dim * rank * para_nums)
    
    def get_mask(self, class_mask, task_id, args, logits):
        if self.mask is None or task_id != self.last_task_id:
            self.last_task_id = task_id
            self.mask = class_mask[task_id]
            # Create a boolean mask for all classes
            # Initialize mask_tensor to True for allowed classes, False otherwise
            self.mask_tensor = torch.full((args.nb_classes,), False, dtype=torch.bool, device=logits.device)
            self.mask_tensor[self.mask] = True  # Set allowed classes to True
    
    def tree_search(self, task_id, device):
        cosine_sim = False
        
        # all_accumulate_grads is a list of tensors: (lora_depth, dim * rank * para_nums) * num_tasks
        if self.all_grad is None:
            self.all_grad = torch.stack(self.all_accumulate_grads[:task_id], dim=0).to(device, non_blocking=True)
            self.all_grad_device = self.all_grad
            
            if cosine_sim:
                # normalize all_grad:
                self.all_grad = self.all_grad / (torch.norm(self.all_grad, dim=2).mean(dim=0) + 1e-5).unsqueeze(0).unsqueeze(2)
            
            # Initialize similarity tensor if not exists
            if self.sim is None:
                self.sim = torch.zeros((task_id, self.all_grad.shape[1]), device=device)
                self.num_of_selected = torch.zeros(self.args.num_tasks, self.all_grad.shape[1]).to(device, non_blocking=True)

        if cosine_sim:
            # We don't calculate all similarities here anymore
            # The similarities will be updated incrementally in update_similarity method
            sim = self.sim.clone()  # Clone to avoid modifying original sim
            
            # Calculate average similarity
            valid_mask = self.num_of_selected[:task_id, :] > 0
            sim[valid_mask] = sim[valid_mask] / self.num_of_selected[:task_id, :][valid_mask]
            
            # add Bandits UCB:
            if self.num_of_selected is not None:
                sim += (1.0 / torch.sqrt(2 * self.num_of_selected[:task_id, :] + 1e-5)
                        * math.sqrt(math.log(2 * self.total_rounds * (self.tmp_rounds + 1) * (self.tmp_rounds + 2))))
        else:
            # For L1 norm distance, we'll use the existing similarity tensor
            sim = self.sim.clone()  # Clone to avoid modifying original sim
            
            # Calculate average similarity
            valid_mask = self.num_of_selected[:task_id, :] > 0
            sim[valid_mask] = sim[valid_mask] / self.num_of_selected[:task_id, :][valid_mask]
            
            # Add Lower Confidence Bound (LCB)
            if self.num_of_selected is not None:
                # Subtract exploration bonus (negative since we're minimizing distance)
                sim -= (1.0 / torch.sqrt(2 * self.num_of_selected[:task_id, :] + 1e-5)
                        * math.sqrt(math.log(2 * self.total_rounds * (self.tmp_rounds + 1) * (self.tmp_rounds + 2))))
            sim = -sim
        
        sim += torch.min(sim)  # Shift to make all similarities positive
        # Searching on Tree Structure
        first_idx = torch.multinomial(torch.softmax(torch.sum(sim, dim=1), dim=0), num_samples=1, replacement=True).item()
        similarity = 1.0
        if self.kd_tree_root is not None and self.kd_tree_root.left is not None:
            if first_idx in self.kd_tree_root.left.task_indices:
                similarity = self.kd_tree_root.left.median_similarity if self.kd_tree_root.left.median_similarity is not None else 1.0
                sim[self.kd_tree_root.left.task_indices] *= min(similarity, 1.5)
            else:
                similarity = self.kd_tree_root.right.median_similarity if self.kd_tree_root.right.median_similarity is not None else 1.0
                sim[self.kd_tree_root.right.task_indices] *= min(similarity, 1.5)
        
        if self.tmp_rounds % 100 == 0:
            print_rank_0('\033[34m****first idx: {}, similarity: {}\033[0m'.format(first_idx, similarity), self.args.global_rank)
        
        sim = sim / (torch.max(sim) - torch.min(sim) + 1e-5)
        sim[task_id:, :] = -torch.inf
        sim_normalized = torch.softmax(sim, dim=0)
        
        # sample from the distribution:
        prev_id_matrix = torch.multinomial(sim_normalized.T, num_samples=1, replacement=True).reshape(-1)
        
        self.num_of_selected[prev_id_matrix, torch.arange(sim.shape[1])] += 1
        
        self.update_similarity(prev_id_matrix, device)
        return prev_id_matrix
    
    def get_loss(self, _grad_current, loss, task_id, prev_id_matrix):
        reg_loss = tree_lora_loss(_grad_current, self.all_grad_device, task_id, prev_id_matrix)
        
        # accelerate:
        reg_loss = reg_loss / (reg_loss.detach().clone() + 1e-5) * loss.detach().clone() * self.tmp_reg
        # reg_loss = reg_loss / -20000
        
        return reg_loss
    
    def update_similarity(self, prev_id_matrix, device):
        """
        Update similarity tensor for selected task indices
        """
        if self.sim is None:
            return
        
        cosine_sim = False
        if cosine_sim:
            # Calculate similarity only for selected tasks
            for depth_idx, prev_id in enumerate(prev_id_matrix):
                self.sim[prev_id, depth_idx] += torch.dot(
                    self.current_grad[depth_idx],
                    self.all_grad[prev_id, depth_idx]
                ).item()
        else:
            # Update L1 distance for selected tasks
            for depth_idx, prev_id in enumerate(prev_id_matrix):
                self.sim[prev_id, depth_idx] -= torch.sum(
                    torch.abs(self.current_grad[depth_idx] - self.all_grad[prev_id, depth_idx])
                ).item()
