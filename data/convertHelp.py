
import json
import csv
import pandas as pd

def wread():
    # 输入文件路径
    input_file_path = 'firefly-train.jsonl'
    # 输出CSV文件路径
    output_csv_path = 'output.csv'

    # 准备写入CSV文件
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入CSV头部
        csv_writer.writerow(['category', 'human', 'assistant'])

        # 逐行读取JSONL文件
        with open(input_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                # 检查kind字段是否为KeywordRecognition
                if data.get('kind') == 'KeywordRecognition':
                    # 提取你需要的字段
                    row = [data.get('kind'), data.get('input'), data.get('target')]
                    csv_writer.writerow(row)

def merage():
    # CSV文件路径
    file_path1 = 'csv/aspcoder_train.csv'
    file_path2 = 'output.csv'

    # 读取CSV文件
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # 合并DataFrame
    merged_df = pd.concat([df1, df2])

    # 输出合并后的DataFrame
    print(merged_df)

    # 保存合并后的DataFrame到新的CSV文件
    output_path = 'merged.csv'
    merged_df.to_csv("aspcoder_train.csv", index=False)
if __name__ == "__main__":
    merage()