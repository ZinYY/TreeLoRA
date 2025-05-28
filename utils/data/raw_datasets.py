# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
import re
import os


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        # default load from disk
        if "Anthropic/hh-rlhf" in dataset_name:
            self.raw_datasets = load_from_disk(dataset_name)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_answer(self, sample):
        return

    def get_prompt_and_answer(self, sample):
        return



# added for Anthropic/hh-rlhf
# adapt for llama2 chat
"""
Data examples:
{
    "chose": " Human: What are some cuss words in english? Assistant: Here’s an incomplete list. Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch, Human: What's your favorite one? Assistant: I haven't even thought about it.",
    "rejected": " Human: What are some cuss words in english? Assistant: Here’s an incomplete list. Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch, Human: What's your favorite one? Assistant: Ass."
}

"""
class AnthropichhrlhfDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)

        self.dataset_name = "Anthropic/hh-rlhf"
        self.dataset_name_clean = "Anthropic_hh_rlhf"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        segments = sample['rejected'].split('Assistant:')
        prompt = "Assitant:".join(segments[:-1])
        return prompt + "Assistant:"

    def get_answer(self, sample):
        segments = sample['rejected'].split('Assistant:')
        rejected = segments[-1]
        return rejected

    def get_prompt_and_answer(self, sample):
        return sample['rejected']



class LocalJsonFileDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, for_backbone=False):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "local_jsonfile"
        self.dataset_name_clean = "jsonfile"
        assert os.path.exists(dataset_name), f"Not found, plz check path {dataset_name}!"
        self.for_backbone = for_backbone
        self.raw_datasets = load_dataset('json',
                                         data_files={
                                             "train":
                                             dataset_name + '/train.json',
                                             "eval":
                                             dataset_name + '/eval.json',
                                             "test":
                                             dataset_name + '/test.json',
                                         })

    def get_train_data(self):
        if self.raw_datasets['train'] is not None:
            return self.raw_datasets['train']
        return None

    def get_eval_data(self):
        if self.raw_datasets['eval'] is not None:
            return self.raw_datasets['eval']
        return None

    def get_test_data(self):
        if self.raw_datasets['test'] is not None:
            return self.raw_datasets['test']
        return None

    def get_prompt(self, sample):
        if sample['prompt'] is not None:
            return sample['prompt']
        return None

    def get_answer(self, sample):
        if sample['answer'] is not None:
            return sample['answer']
        return ''

    def get_prompt_and_answer(self, sample):
        if sample['prompt'] is not None and sample['answer'] is not None:
            return sample['prompt'] + "\n" + sample['answer']
        return None
