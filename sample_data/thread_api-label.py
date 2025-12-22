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


client = OpenAI(api_key="sk-", base_url="")

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



def send_request(prompt, model="qwen-max"): 
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
            reasoning_content = ""  
            answer_content = ""     
            is_answering = False   


            completion = client.chat.completions.create(
                model=model,  
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            )

            for chunk in completion:
                if not chunk.choices:
                    print("\nUsage:")
                    print(chunk.usage)
                else:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                        reasoning_content += delta.reasoning_content
                    else:
                        if delta.content != "" and is_answering is False:
                            is_answering = True
                        answer_content += delta.content
            return answer_content
    except Exception as e:
        print(f"{model} API Failed: {e}")
        return None

def split_solution(solution, max_words_per_chunk=4096, min_last_chunk_words=2048):
    words = solution.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(current_chunk) + 1 <= max_words_per_chunk:
            current_chunk.append(word)
        else:
            # If adding the new word would exceed the maximum word count
            if len(current_chunk) < min_last_chunk_words:
                # If the current chunk is smaller than the minimum, try merging it with the previous chunk
                if chunks:
                    chunks[-1].extend(current_chunk)
                    current_chunk = []
            else:
                # Check whether the last sentence in the current chunk can be kept intact
                if current_chunk and current_chunk[-1].endswith(('.', '?', '!')):
                    # If the last word ends a complete sentence, finalize the current chunk
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                else:
                    # If not, append the word and check again
                    current_chunk.append(word)
                    # Verify whether the sentence is now complete
                    if current_chunk and current_chunk[-1].endswith(('.', '?', '!')):
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []

    if current_chunk:
        # Append any remaining words as the final chunk
        chunks.append(" ".join(current_chunk))

    return chunks

def process_point_item(data, idx, output_file):
    source = data['source']
    correct = data["correct"]
    response = data['response']
    question = data['question']
    ans = data["ans"]
    answer = data["label_answer"]
    
    
    
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
    # ajust max_workers according to api
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, item in enumerate(data_list):
            futures.append(executor.submit(process_point_item, item, idx, output_file))

        with tqdm(total=len(futures), desc="processing") as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

def generate(input_file, output_file):

    existed = set() 

    if not os.path.exists(output_file):  
        with open(output_file, 'w') as f:  
            f.write("")  
        print(f"created {output_file}")  

    else:
        with open(output_file, 'r', encoding='utf-8') as f:
            if not f.readline().strip() == '':
                for line in f:
                    item = json.loads(line)
                    if item["labeled_solution"] != "":
                        existed.add(item['question'])
            print(f"existed:{len(existed)}")
    
    data_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if not d["question"] in existed:
                if 'source' in d.keys():
                    if d['response'] != "":
                        data_list.append(d)
                else:
                    data_list.append(d)
    print("data list: ",len(data_list))

    batch_process(data_list, output_file)



input_file = 'input.jsonl'
output_file = 'label-output.jsonl'
generate(input_file,output_file)
