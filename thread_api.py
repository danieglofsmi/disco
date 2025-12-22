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


PROMPT = """The following is the reasoning chain that is used to answer a difficult math problem. Please process the reasoning chain according to the following rules:
1. Label all segments that are potentially final results in the reasoning chain with \\possibleAnswer{{}} format. DO NOT label all the possible intermediate results, ONLY label the ones that could be the final answers, no matter it's correct or wrong. Label as many as you could.
2. An example of the \\possibleAnswer{{}} annotation: "Wait, 5 times 360 is 1800, and 1800 divided by 36. Let's do that division: 1800 ÷ 36. Hmm, 36 times 50 is 1800, right? Because 36 x 50 is 1800. So, 1800 ÷ 36 = \\possibleAnswer{{50}}. Therefore, the degrees for cherry pie would be \\possibleAnswer{{50}} degrees."
3. Label all segments that indicate a shift in reasoning within the text reasoning chain using the \\thoughtchange{{}} format. Label as many as you could.
4.  An example of the \\thoughtchange{{}} annotation: "\\thoughtchange{{Wait, maybe}} I messed up the dailyprogress.\n\n\\thoughtchange{{Wait, hold on}}. If the original totaltime is T days, then when they switch to the newequipment after 1/3 of the tunnel is done, whichtook T/3 days, and then the remaining 2/3 is doneat a slower daily rate"  
5. DO NOT change other parts and keep them exactly the same as the the original solution.

### original solution:
{solution}

### Result:"""



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

def split_solution(solution, max_words_per_chunk=4096, min_last_chunk_words=2048):
    words = solution.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(current_chunk) + 1 <= max_words_per_chunk:
            current_chunk.append(word)
        else:
            # 如果当前块的单词数加上新单词的单词数超过了最大单词数
            if len(current_chunk) < min_last_chunk_words:
                # 如果当前块的单词数小于最小块大小，尝试合并到上一个块
                if chunks:
                    chunks[-1].extend(current_chunk)
                    current_chunk = []
            else:
                # 检查是否可以将当前块的最后一个句子完整地保留
                if current_chunk and current_chunk[-1].endswith(('.', '?', '!')):
                    # 如果最后一个单词是一个完整的句子，则直接添加当前块
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                else:
                    # 如果最后一个单词不是一个完整的句子，则尝试将新单词加入当前块
                    current_chunk.append(word)
                    # 检查是否可以将当前块的最后一个句子完整地保留
                    if current_chunk and current_chunk[-1].endswith(('.', '?', '!')):
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []

    if current_chunk:
        # 如果最后一个块不为空，直接添加
        chunks.append(" ".join(current_chunk))

    return chunks

def process_point_item(data, idx, output_file):
    source = data['source']
    # source = 'amc2023'
    correct = data["correct"]
    response = data['response']
    question = data['question']
    ans = data["ans"]
    answer = data["label_answer"]
    
    
    # print(f"{idx} executing: {data['uuid']}")
    # if solution.split() > 16384:
    #     print(f"{uuid} too long")
    #     continue
    
    max_retries = 5
    response_parts = []
    for _ in range(max_retries):
        solution_chunks = split_solution(response)
        for chunk in solution_chunks:
            for _ in range(max_retries):
                chunk_response = send_request(PROMPT.format(solution=chunk))
                if chunk_response is not None:
                    response_parts.append(chunk_response)
                    break
                else:
                    time.sleep(10)
        if response_parts:
            break

    label_response = "\n".join(response_parts)

    meta = {'source':source, 'correct':correct, 'ans':ans, 'answer': answer,'question': question,  'labeled_solution': label_response}
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
                    if item["labeled_solution"] != "":
                        existed.add(item['question'])
            print(f"existed:{len(existed)}")
    
    # 读取整个数据集为列表
    data_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if not d["question"] in existed:
                if 'source' in d.keys():
                    if  d['source'] == 'amc2023':
                        data_list.append(d)
                else:
                    data_list.append(d)
    print("data list: ",len(data_list))

    batch_process(data_list, output_file)

input_file = '7b-initial-new/7b-grpo-initial-step50-hf_test100.jsonl'
output_file = '7b-initial-new/label-7b-grpo-initial-step50-hf_amc2023.jsonl'
generate(input_file,output_file)

# input_file = '7b-base/7b-base_amc2023.jsonl'
# output_file = '7b-base/label-7b-base_amc2023.jsonl'
# generate(input_file,output_file)

# input_file = '7b-grpo-base/7b-grpo-base_amc2023.jsonl'
# output_file = '7b-grpo-base/label-7b-grpo-base_amc2023.jsonl'
# generate(input_file,output_file)

# input_file = '7b-initial-new/7b-grpo-initial-step100-hf_amc2023.jsonl'
# output_file = '7b-initial-new/label-7b-grpo-initial-step100-hf_amc2023.jsonl'
# generate(input_file,output_file)