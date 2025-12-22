# -*- encoding:utf-8 -*-
import random
import json
from openai import OpenAI
from tqdm import tqdm
import time
import concurrent.futures
import threading
import itertools
import sys
import os
from verify import last_boxed_only_string,remove_boxed_or_possibleAnswer,is_correct

# GPT
# client = OpenAI(api_key="sk-LWzY4DovAFCgyNdgr1zgLYVvy1Moh3ErL6jwNur5jcs2jqRY", base_url="https://api.chatanywhere.tech/v1")
# BaiLian
# client = OpenAI(api_key="sk-d26f44789140496aa13d4e01def9d5c4", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
#  sk-fbc6bcfdb0ba4e3abe159906f0d3f798
# client = OpenAI(api_key="sk-2403a97b97e8498bb1b4e9accf1fa7c7", base_url="https://api.deepseek.com")
# 'gpt-4o'
# client = OpenAI(api_key="sk-LWzY4DovAFCgyNdgr1zgLYVvy1Moh3ErL6jwNur5jcs2jqRY", base_url="https://api.chatanywhere.tech/v1")

# client = OpenAI(api_key="sk-d26f44789140496aa13d4e01def9d5c4", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# BaiLian
client = OpenAI(api_key="sk-bc238a1af6f44edd83b427a10bcdd3a5", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
# sk-99ecdb4dc2394a21abb959b6036c8d66

write_lock = threading.Lock()


PROMPT = """<｜User｜>Try to solve the following question step by step. If the final answer is obtained, use \\boxed{{}} to represent it.
### Question:{question}
<｜Assistant｜><think>\n"""

def read_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)  # 解析每一行
            data.append(record)
    return data

def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def send_request(prompt, model="qwen-plus-latest"): 
    """发送请求到 OpenAI API 并返回结果（线程安全）"""
    try:
        if model != 'qwq-plus':
            response = client.chat.completions.create(
                model=model,    #'deepseek-chat'
                messages=[
                    {"role": "user", "content": prompt}
                ],

                max_completion_tokens=16384
            )
            return response.choices[0].message.content if response.choices else None
        else:
            reasoning_content = ""  # 定义完整思考过程
            answer_content = ""     # 定义完整回复
            is_answering = False   # 判断是否结束思考过程并开始回复

            # 创建聊天完成请求
            completion = client.chat.completions.create(
                model=model,  # 此处以 qwq-32b 为例，可按需更换模型名称
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            )

            for chunk in completion:
                # 如果chunk.choices为空，则打印usage
                if not chunk.choices:
                    print("\nUsage:")
                    print(chunk.usage)
                else:
                    delta = chunk.choices[0].delta
                    # 打印思考过程
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                        reasoning_content += delta.reasoning_content
                    else:
                        # 开始回复
                        if delta.content != "" and is_answering is False:
                            is_answering = True
                        # 打印回复过程
                        answer_content += delta.content
            return answer_content
    except Exception as e:
        print(f"{model} API 请求失败: {e}")
        return None


def process_point_item(data, idx, output_file):
    source = data['source']
    question = data['question']
    label_answer = data["answer"]

    
    max_retries = 5
    for _ in range(max_retries):
        response = send_request(PROMPT.format(question=question))
        if response is not None:
            ans = last_boxed_only_string(response)
            if ans:
                ans = remove_boxed_or_possibleAnswer(ans)
                correct = is_correct(ans,label_answer)
            else:
                correct = False  
            break
        else:
            time.sleep(5)

    meta = {'source':source, 'correct':correct, "label_answer": label_answer, 'ans':ans, 'question': question,  "response": response}
    with write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
            

def batch_process(data_list, output_file):
    """批量处理JSON数据（多线程版本）"""
    # 创建线程池（根据API限流调整workers数量）
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, item in enumerate(data_list):
            futures.append(executor.submit(process_point_item, item, idx, output_file))

        # 使用进度条监控处理进度
        with tqdm(total=len(futures), desc="处理进度") as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

def generate(input_file, output_file):
    # if len(sys.argv) < 3:
    #     print("请提供输入的JSONL文件路径和评估模型名称。")
    #     sys.exit(1)
        
    # input_file = sys.argv[1]
    # # print(input_file)
    # evaluator = sys.argv[2]
    # lang = sys.argv[3]
    # output_file = evaluator + '_pointwise_' + input_file.split('/')[-1]
    # print(output_file)

    # input_file = 'D:/BIT/NLP/datasets/OpenR1-Math-220k-train-00000-of-00010.jsonl'

    existed = set() # 当程序崩溃时,使数据能够断点续传,避免重复生成

    if not os.path.exists(output_file):  
        with open(output_file, 'w') as f:  
            f.write("")  
        print(f"{output_file} 不存在，已创建该文件。")  

    else:
        with open(output_file, 'r', encoding='utf-8') as f:
            if not f.readline().strip() == '':
                for line in f:
                    item = json.loads(line)
                    if item["response"] != "":
                        existed.add(item['question'])
            print(f"existed:{len(existed)}")
    
    # 读取整个数据集为列表
    data_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if not d['question'] in existed:
                data_list.append(d)
    print("data list: ",len(data_list))

    batch_process(data_list, output_file)


input_file = 'testset/merge_test.jsonl'
output_file = 'qwen3_infrence.jsonl'
generate(input_file,output_file)