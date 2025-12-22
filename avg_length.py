# import json
# def read_jsonl(jsonl_file):
#     data = []
#     with open(jsonl_file, "r", encoding="utf-8") as file:
#         for line in file:
#             record = json.loads(line)  # 解析每一行
#             data.append(record)
#     return data

# file_base = """7b-grpo-base/7b-grpo-base_{dataset}.jsonl"""
# data_ls=['aime','amc2023','gpqa','gpqa_diamond','gsm8k','gsm8k_test','math','math500','TheoremQA','valid27','valid100']
# for dataset in data_ls:
#     try:
#         data = read_jsonl(file_base.format(dataset=dataset))
#         length=[]
#         for d in data:
#             if d['ans']:
#                 length.append(len(d['response']))
#         print(dataset, sum(length)/len(length))
#     except:
#         pass

import json
from collections import defaultdict

# 输入的 JSONL 文件路径
input_file = "7b-initial-new/7b-grpo-initial-step100-hf_test100.jsonl"

# 创建一个字典来存储每个 source 的 response 长度总和和数量
source_stats = defaultdict(lambda: {"total_length": 0, "count": 0})

# 逐行读取 JSONL 文件
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        # 解析每一行的 JSON 数据
        data = json.loads(line)
        
        # 提取 source 和 response
        if data['ans']:
            source = data.get("source")
            response = data.get("response")
            
            # 如果 source 和 response 都存在
            if source and response:
                # 累加 response 的长度
                source_stats[source]["total_length"] += len(response)
                # 增加计数
                source_stats[source]["count"] += 1

# 计算每个 source 的平均 response 长度
average_lengths = {}
for source, stats in source_stats.items():
    if stats["count"] > 0:
        average_lengths[source] = stats["total_length"] / stats["count"]
    else:
        average_lengths[source] = 0

# 输出结果
for source, avg_length in average_lengths.items():
    print(f"{source} {avg_length:.2f}")