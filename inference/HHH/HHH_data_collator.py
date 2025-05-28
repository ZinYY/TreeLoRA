import logging
import torch
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True  # ‘longest’
    max_prompt_len: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 1
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        model_inputs = self.decoder_call(batch, self.return_tensors)

        return model_inputs

    # only support left padding for now
    def tokenize(self, sentence, cutoff_len, add_bos_token=True, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            sentence,
            truncation=True,
            max_length=cutoff_len,
            add_special_tokens=False,
            padding=False,
            return_tensors=None,
        )

        if (
                len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if (
                len(result["input_ids"]) < cutoff_len
                and add_bos_token
        ):
            result["input_ids"] = [self.tokenizer.bos_token_id] + result["input_ids"]
            result["attention_mask"] = [1] + result["attention_mask"]

        return result

    # support decoder-only models for left padding
    def decoder_call(self, batch, return_tensors):
        # to fix the bug
        sources = []
        tokenized_sources = []
        actual_max_len = 0
        limit_len = self.max_prompt_len

        for instance in batch:
            instruction = instance['prompt']
            sources.append(instruction)
            tokenize_source = self.tokenize(instruction, limit_len, add_bos_token=True, add_eos_token=False)
            tokenized_sources.append(tokenize_source)

            if len(tokenize_source["input_ids"]) > actual_max_len:
                actual_max_len = len(tokenize_source["input_ids"])

        actual_pad_len = (
                    (actual_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of)

        for idx in range(len(tokenized_sources)):
            pad_len = actual_pad_len - len(tokenized_sources[idx]["input_ids"])
            assert sum(tokenized_sources[idx]["attention_mask"]) == len(tokenized_sources[idx]["input_ids"])
            tokenized_sources[idx]["input_ids"] = [self.tokenizer.pad_token_id] * pad_len + tokenized_sources[idx][
                "input_ids"]

            tokenized_sources[idx]["attention_mask"] = [0] * pad_len + tokenized_sources[idx]["attention_mask"]


        model_inputs = {'input_ids': torch.tensor([source["input_ids"] for source in tokenized_sources]),
                        'attention_mask': torch.tensor([source["attention_mask"] for source in tokenized_sources])}

        model_inputs['sources'] = sources

        return model_inputs
