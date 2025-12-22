import json
# def read_jsonl(jsonl_file):
#     data = []
#     with open(jsonl_file, "r", encoding="utf-8") as file:
#         for line in file:
#             record = json.loads(line)  # 解析每一行
#             data.append(record)
#     return data

# def write_jsonl(data, jsonl_file):
#     with open(jsonl_file, "w", encoding="utf-8") as file:
#         for record in data:
#             json.dump(record, file, ensure_ascii=False)  # 将字典转换为JSON字符串
#             file.write("\n")

# jsonl_file = "/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/7b-initial-120/7b-initial-120_aime.jsonl"
# new_jsonl_file = "/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/7b-initial-120/7b-initial-120_aime_1.jsonl"
# new_dataset = []
# data = read_jsonl(jsonl_file)
# for d in data:
#     if 'error' in d.keys():
#         continue
#     else:
#         new_dataset.append(d)
# write_jsonl(new_dataset, new_jsonl_file)


def remove_annotations(input_string):
    patterns = ["\\possibleAnswer{", "\\thoughtchange{"]
    
    for pattern in patterns:
        while pattern in input_string:
            start = input_string.find(pattern)
            if start == -1:
                break  # 如果没有找到模式，退出循环

            # 找到匹配的右大括号位置
            bracket_count = 1
            end = start + len(pattern)
            while end < len(input_string) and bracket_count > 0:
                if input_string[end] == '{':
                    bracket_count += 1
                elif input_string[end] == '}':
                    bracket_count -= 1
                end += 1

            # 如果没有找到匹配的右大括号，退出循环
            if bracket_count != 0:
                break

            # 移除标注部分
            content_start = start + len(pattern)
            content_end = end - 1
            input_string = input_string[:start] + input_string[content_start:content_end] + input_string[end:]

    return input_string


prompt = """Try to solve the following question step by step. If the final answer is obtained, use \\boxed{{}} to represent it.
### Question:{question}"""

with open('sft_train_lxy.json','r',encoding='utf-8') as f:
    data1 = json.load(f)
with open('sft_test_lxy.json','r',encoding='utf-8') as f:
    data2 = json.load(f)
data = data1+data2

new_ls = []
for d in data:
    tmp = dict()
    question = d['instruction'].split("### Question:")[-1].strip()
    tmp['instruction'] = prompt.format(question=question)
    tmp['input'] = ""
    tmp['output'] = remove_annotations(d['output'])
    tmp['system'] = ""
    tmp['history'] = []
    new_ls.append(tmp)
with open('sft_train_base.json','w',encoding='utf-8') as f:
    json.dump(new_ls, f, ensure_ascii=False,indent=4)