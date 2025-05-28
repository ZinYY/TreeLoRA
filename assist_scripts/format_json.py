# import json


# def transform_json(input_path, output_path):
#     # Read the input JSON file
#     with open(input_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     # Transform the data
#     transformed_data = []
#     for item in data:
#         transformed_item = {
#             'prompt': item['sentence'],
#             'answer': item['label']
#         }
#         transformed_data.append(transformed_item)

#     # Write the transformed data to output file
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(transformed_data, f, ensure_ascii=False, indent=2)


# if __name__ == '__main__':
#     input_path = r"E:\TreeLoRA\Rebuttal\O-LoRA\O-LoRA-main\O-LoRA-main\CL_Benchmark\BoolQA\BoolQA\dev.json"  # Hard-coded input path
#     output_path = r"E:\TreeLoRA\Rebuttal\O-LoRA\O-LoRA-main\O-LoRA-main\CL_Benchmark\BoolQA\BoolQA\dev_process.json"  # Hard-coded output path
#     transform_json(input_path, output_path)
    
import json
import os
import shutil
from pathlib import Path


def transform_json(input_path, output_path):
    # Read the input JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Transform the data
    transformed_data = []
    for item in data:
        transformed_item = {
            'prompt': item['sentence'],
            'answer': item['label']
        }
        transformed_data.append(transformed_item)

    # Write the transformed data to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)


def process_directory(directory_path):
    # Convert to Path object for easier path handling
    dir_path = Path(directory_path)
    
    # Expected files
    expected_files = {'dev.json', 'train.json', 'test.json', 'labels.json'}
    
    # Get all JSON files in directory
    json_files = {f.name for f in dir_path.glob('*.json')}
    
    # Check if directory contains exactly the expected files
    if json_files != expected_files:
        missing_files = expected_files - json_files
        extra_files = json_files - expected_files
        error_msg = []
        if missing_files:
            error_msg.append(f"Missing files: {', '.join(missing_files)}")
        if extra_files:
            error_msg.append(f"Unexpected files: {', '.join(extra_files)}")
        raise ValueError(f"Directory does not contain exactly the expected files.\n" + "\n".join(error_msg))
    
    # Process files except label.json
    for json_file in dir_path.glob('*.json'):
        if json_file.name == 'labels.json':
            continue
            
        # Create backup of original file
        backup_path = json_file.parent / f'raw_{json_file.name}'
        shutil.copy2(json_file, backup_path)
        
        # Determine output filename
        output_name = 'eval.json' if json_file.name == 'dev.json' else json_file.name
        output_path = json_file.parent / output_name
        
        # Process the file
        transform_json(str(json_file), str(output_path))
        print(f'Processed: {json_file.name} -> {output_name}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process JSON files in a directory')
    parser.add_argument('directory_path', type=str, help='Path to directory containing JSON files')
    args = parser.parse_args()
    
    try:
        process_directory(args.directory_path)
        print("All files processed successfully!")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)