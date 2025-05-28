# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import time

import math
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import LlamaForCausalLM, LlamaConfig

from utils.utils import print_rank_0


class TIKTOK:
    def __init__(self, args):
        self.t0 = time.time()
        self.all_module_time = {}
        self.args = args
        self.nums = 0
    
    def tik(self):
        self.t0 = time.time()
    
    def tok(self, text):
        self.nums += 1
        spt = time.time() - self.t0
        # print green:
        if self.args.print_tiktok:
            print_rank_0("\033[0;32m" + str(text) + ":" + str(spt) + "\033[0m")
        
        if text not in self.all_module_time:
            self.all_module_time[text] = 0.0
        self.all_module_time[text] += spt
    
    def print_time(self, rank=0):
        for key in self.all_module_time.keys():
            print_rank_0("\033[0;32m" + key + ":" + str(self.all_module_time[key] / self.nums) + "\033[0m", rank)


def replace_silu_with_relu(model):
    """
    Recursively replace all SiLU activation functions with ReLU in the model.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.SiLU):
            setattr(model, name, nn.ReLU())
        else:
            replace_silu_with_relu(module)


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=True)

    # llama use eos_token_id but not end_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    # compatible with OPT and llama2
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model
