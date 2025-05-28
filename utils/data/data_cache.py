import os
import datasets
from datasets import load_dataset

DATA_PATH = "./DATA_cache"


ds_name = "Anthropic/hh-rlhf"
dataset = load_dataset("Anthropic/hh-rlhf")
out_path = os.path.join(DATA_PATH, ds_name)
dataset.save_to_disk(out_path)


dataset = datasets.load_from_disk(out_path)
