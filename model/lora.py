import json

from model.base_model import CL_Base_Model
import os
import time
from utils.utils import print_rank_0, save_hf_format, save_zero_three_model


class lora(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
    
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
        
        # if self.args.output_dir is not None:
        #     print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)
        #
        # if self.args.global_rank == 0:
        #     save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(round))
        #
        # if self.args.zero_stage == 3:
        #     # For zero stage 3, each gpu only has a part of the model, so we need a special save function
        #     save_zero_three_model(self.model,
        #                           self.args.global_rank,
        #                           self.args.output_dir,
        #                           zero_stage=self.args.zero_stage,
        #                           sub_folder=str(round))
        # print_rank_0('Successfully saving model after round {}'.format(round), self.args.global_rank)
