import csv
import json
import os
import pandas as pd
from html import unescape

           
"""
将json文件转换为csv文件
"""
def __from_json_to_csv(train_file_path,train_file_name):
     # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data from file
    
    # Open the CSV file for writing
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(['category','human', 'assistant'])
        
        # Write data rows
        for entry in data:
            instruction = entry.get('instruction', '')  # Get 'instruction' or default to empty string
            output = entry.get('output', '')  # Get 'output' or default to empty string
            writer.writerow(['general',instruction, output])    
           
"""
将csv文件转换为json文件
"""
def from_csv_to_jsonl(train_file_path,train_file_name):
    csv_file_path=train_file_path+"/csv/"+train_file_name
    jsonl_file_name=train_file_name.replace('.csv','.jsonl')
    jsonl_file_path=train_file_path+"/"+jsonl_file_name
    # 检查是否有重名的文件存在，如果有，则删除
    if os.path.exists(jsonl_file_path):
        os.remove(jsonl_file_path)
    # 初始化conversation_id
    conversation_id = 1
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        # 读取CSV文件
        reader = csv.DictReader(csvfile)
        with open(jsonl_file_path, mode='w', encoding='utf-8') as jsonlfile:
            for row in reader:
                conversation_id += 1  # conversation_id自增
                data = {
                    "conversation_id": conversation_id,
                    "category": row['category'],
                    "conversation": [
                        {"human": row['human'], "assistant": row['assistant']}
                    ],
                    "dataset": "aspcoder" 
                }
                # 将字典转换为JSON字符串，并写入文件
                jsonlfile.write(json.dumps(data, ensure_ascii=False) + '\n')
    return jsonl_file_path

"""
将excel文件转换为json文件
"""
def from_csv_to_jsonl(train_file_path,train_file_name):
    csv_file_path=train_file_path+"/csv/"+train_file_name
    jsonl_file_name=train_file_name.replace('.csv','.jsonl')
    jsonl_file_path=train_file_path+"/"+jsonl_file_name
    # 检查是否有重名的文件存在，如果有，则删除
    if os.path.exists(jsonl_file_path):
        os.remove(jsonl_file_path)
    # 初始化conversation_id
    conversation_id = 1
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        # 读取CSV文件
        reader = csv.DictReader(csvfile)
        with open(jsonl_file_path, mode='w', encoding='utf-8') as jsonlfile:
            for row in reader:
                # 对HTML内容进行解码
                human =row['human']
                types = row['category']
                if types == "jra":
                    human="异常鉴别："+human
                if types == "know":
                    human="要素识别："+human
                if types == "KeywordRecognition":
                    human="关键词识别："+human
                conversation_id += 1  # conversation_id自增
                data = {
                    "conversation_id": conversation_id,
                    "category": row['category'],
                    "conversation": [
                        {"human": human, "assistant": row['assistant']}
                    ],
                    "dataset": "aspcoder" 
                }
                # 将字典转换为JSON字符串，并写入文件
                jsonlfile.write(json.dumps(data, ensure_ascii=False) + '\n')
    return jsonl_file_path

# Execute the function
if __name__ == "__main__":
    json_to_csv(json_file_path, csv_file_path)
