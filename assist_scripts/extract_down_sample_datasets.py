import os
import json
import random
import shutil
from pathlib import Path

# Sampling configurations
SAMPLE_CONFIGS = {
    'Small': {'train': 500, 'test': 100},
    'Mid'  : {'train': 1000, 'test': 500},
    'Big'  : {'train': 5000, 'test': 1000}
}

TASK_PROMT = {
    "yelp"   : "What is the sentiment of the following paragraph? Choose one from the option.\nOption: very negative, negative, neutral, positive, very positive \n",
    "amazon" : "What is the sentiment of the following paragraph? Choose one from the option.\nOption: very negative, negative, neutral, positive, very positive \n",
    "dbpedia": "What is the topic of the following paragraph? Choose one from the option.\nOption: Company, Educational Institution, Artist, Athlete, Office Holder, Mean of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film, Written Work \n",
    "yahoo"  : "What is the topic of the following paragraph? Choose one from the option.\nOption: Society & Culture, Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships, Politics & Government \n",
    "agnews" : "What is the topic of the following paragraph? Choose one from the option.\nOption: World, Sports, Business, Science or Technology \n",
    "MNLI"   : "What is the logical relationship between the \"sentence 1\" and the \"sentence 2\"? Choose one from the option.\nOption: neutral, entailment, contradiction \n",
    "QQP"    : "Whether the \"first sentence\" and the \"second sentence\" have the same meaning? Choose one from the option.\nOption: False, True \n",
    "RTE"    : "What is the logical relationship between the \"sentence 1\" and the \"sentence 2\"? Choose one from the option.\nOption: contradiction, entailment \n",
    "SST-2"  : "What is the sentiment of the following paragraph? Choose one from the option.\nOption: Good, Bad \n",
    "WiC"    : "Given a word and two sentences, whether the word is used with the same sense in both sentence? Choose one from the option.\nOption: True, False \n",
    "CB"     : "What is the logical relationship between the \"sentence 1\" and the \"sentence 2\"? Choose one from the option.\nOption: entailment, contradiction, neutral \n",
    "COPA"   : "",
    "BoolQA" : "According to the following passage, is the question true or false? Choose one from the option.\nOption: True, False \n",
    "MultiRC": "According to the following passage and question, is the candidate answer true or false? Choose one from the option.\nOption: False, True \n",
    "IMDB"   : "What is the sentiment of the following paragraph? Choose one from the option.\nOption: Good, Bad \n",
    "FOMC"       : "What is the monetary policy stance for the following text? A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\n",
    "NIL"    : "What is the logical relationship between the \"sentence 1\" and the \"sentence 2\"? Choose one from the option.\nOption: neutral, entailment, contradiction \n",
    "C-STANCE"   : "判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n",
    "ScienceQA"  : "Choose an answer for the following question and give your reasons.\n\n",
    "NumGLUE-cm" : "Solve the following math problem.\n",
    "NumGLUE-ds" : "Solve the following math problem.\n",
    "MeetingBank": "Write a summary of the following meeting transcripts.\n",
    "Py150"      : "Continue writing the code.\n",
    "20Minuten"  : "Provide a simplified version of the following paragraph in German.\n\n",
}

END_PROMT = {
    "yelp"   : "Answer:",
    "amazon" : "Answer:",
    "dbpedia": "Answer:",
    "yahoo"  : "Answer:",
    "agnews" : "Answer:",
    "MNLI"   : "Answer:",
    "QQP"    : "Answer:",
    "RTE"    : "Answer:",
    "SST-2"  : "Answer:",
    "WiC"    : "Answer:",
    "CB"     : "Answer:",
    "COPA"   : "Answer:",
    "BoolQA" : "Answer:",
    "MultiRC": "Answer:",
    "IMDB"   : "Answer:",
    "FOMC"   : "Answer:",
}


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def random_sample(data, n):
    """Randomly sample n items from data. If data has fewer items than n, return all data"""
    if len(data) <= n:
        return data
    return random.sample(data, n)


def get_task_name(file_path):
    """Extract task name from file path"""
    path_parts = Path(file_path).parts
    for task_name in TASK_PROMT.keys():
        if task_name in path_parts:
            return task_name
    return None


def add_task_prompt(data, task_name):
    """Add task prompt and end prompt to each item in the data"""
    if not task_name or task_name not in TASK_PROMT:
        return data
    
    task_prompt = TASK_PROMT[task_name]
    # Get end prompt if available
    end_prompt = END_PROMT.get(task_name, '')
    
    for item in data:
        if 'prompt' in item:
            if item['prompt'].endswith("\n"):
                item['prompt'] = task_prompt + item['prompt'] + end_prompt
            else:
                item['prompt'] = task_prompt + item['prompt'] + "\n" + end_prompt
            item['prompt'] = item['prompt'].replace("Answer: ", "").replace("Answer:Answer:", "Answer:").replace(task_prompt + task_prompt, task_prompt)
            item['prompt'] = item['prompt'].replace("Answer:\nAnswer:", "Answer:")
            # replace "\\n" with "\n":
            item['prompt'] = item['prompt'].replace("\\n", "\n")
    return data


def process_dataset(root_dir):
    # Find all json files
    for file_path in Path(root_dir).rglob('*.json'):
        file_name = file_path.name
        
        # Handle labels.json - copy directly
        if file_name == 'labels.json':
            for size_name in SAMPLE_CONFIGS.keys():
                target_path = os.path.join(root_dir, size_name, os.path.relpath(str(file_path), root_dir))
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(file_path, target_path)
                print(f"Copied {file_path} -> {target_path}")
            continue
        
        if file_name not in ['train.json', 'test.json']:
            continue
        
        # Determine if it's train or test
        dataset_type = 'train' if 'train' in file_name else 'test'
        
        # Load original data
        data = load_json(str(file_path))
        
        # Get task name and add task prompt
        task_name = get_task_name(str(file_path))
        
        # Sample for each target size
        for size_name, size_config in SAMPLE_CONFIGS.items():
            # Calculate target path
            relative_path = os.path.relpath(str(file_path), root_dir)
            target_path = os.path.join(root_dir, size_name, relative_path)
            
            # Sample and add task prompt
            sampled_data = random_sample(data, size_config[dataset_type])
            sampled_data = add_task_prompt(sampled_data, task_name)
            
            # Save
            save_json(sampled_data, target_path)
            print(task_name)
            print(f"Processed {file_path} -> {target_path} ({len(sampled_data)} samples)")
            
            # If processing test.json, also create eval.json with the same content
            if dataset_type == 'test':
                eval_path = os.path.join(os.path.dirname(target_path), 'eval.json')
                save_json(sampled_data, eval_path)
                print(f"Created {eval_path} (copy of test data)")


if __name__ == "__main__":
    root_dir = "./data/LLM-CL-Benchmark/raw_O_LoRA_Dataset"
    random.seed(42)  # Set random seed for reproducibility
    process_dataset(root_dir)
