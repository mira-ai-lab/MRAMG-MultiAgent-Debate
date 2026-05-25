import json

academic_dataset = ["arxiv"]
lifestyle_dataset = ["mannual", "recipe"]
web_dataset = ["web", "wiki", "wit"]

input_folder_path = "/data2/qn/MRAMG/outputs/gpt-4o-mini/v2"

# 读一个文件夹下所有的jsonl文件
import os
jsonl_files = [f for f in os.listdir(input_folder_path) if f.endswith(".jsonl")]

jsonl_file_names = [f.split(".")[0] for f in jsonl_files]