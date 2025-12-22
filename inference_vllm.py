import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import sys
import re
from verify import last_boxed_only_string,remove_boxed_or_possibleAnswer,is_correct


model_path = ""
# sampling_params = SamplingParams(max_tokens=16384, temperature=0.0, repetition_penalty=1.50)
sampling_params = SamplingParams(max_tokens=16384, temperature=0.0)
model = LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True, gpu_memory_utilization=0.9)   # bf16
# model = LLM(model=model_path, tensor_parallel_size=4, trust_remote_code=True, dtype="half") # float16 
tokenizer = AutoTokenizer.from_pretrained(model_path)


def make_prompt(question):
    prompt = """You are a helpful assistant. Try to solve the following question step by step. First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> tags, i.e., <think> reasoning process here </think>\n. If the final answer is obtained, use \\boxed{{}} to represent it. 
    ### Question:{question}
    ### Assistant: <think>\n"""

    USER_PROMPT = prompt.format(question=question)

    return [
        {"role": "user", "content": USER_PROMPT}
    ]


def generate_response(question):
    prompt = make_prompt(question)
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
    output = model.generate(prompt, sampling_params)

    return(output[0].outputs[0].text)

def generate_all(input_file, max_data,output_file):
    prompts = []
    data = []
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in open(input_file))
        if max_data:
            total=min(max_data, total_lines)
        else:
            total = total_lines

        for idx, line in tqdm(enumerate(f), total=total, desc="Processing input file"):
            if max_data and idx >= max_data:
                break

            try:
                obj = json.loads(line.strip()) 
                prompt = make_prompt(obj["question"])
                prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
                prompts.append(prompt)
                data.append(obj)

            except json.JSONDecodeError:
                print(f"Error decoding JSON at line {idx+1}")
                print(idx,line)
                continue
            
            
    print("total:",len(prompts))
    outputs = model.generate(prompts, sampling_params)

    # responses = []

    for i, output in enumerate(outputs):
        response = output.outputs[0].text.strip()
        ans = last_boxed_only_string(response)
        if ans:
            ans = remove_boxed_or_possibleAnswer(ans)
            correct = is_correct(ans,data[i]['answer'])
        else:
            correct = False
        meta = {
            "source": data[i]['source'],
            "correct": correct,
            "ans":ans,
            "label_answer": data[i]['answer'],
            "question": data[i]['question'],
            "response": response,
        }

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        


dataset_path = 'test.jsonl'
output_file = "output.jsonl"
generate_all(dataset_path,  max_data=None, output_file=output_file)

