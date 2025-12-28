import json
import os
import pandas as pd
from datasets import Dataset
import random

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

GPQA_PROMPT = """{question}
### Options: 
    (A) {option1}
    (B) {option2}
    (C) {option3}
    (D) {option4} 
Please choose the correct option. 
"""


def make_gpqa_quesiton(data):
    options = ['A', 'B', 'C', 'D']
    correct_position = random.randint(0, 3)
    label_answer = options[correct_position]

    options = [data["Incorrect Answer 1"], data["Incorrect Answer 2"], data["Incorrect Answer 3"], data["Correct Answer"]]

    options.insert(correct_position, options.pop())

    question = GPQA_PROMPT.format(
        question=data["Question"],
        option1=options[0],
        option2=options[1],
        option3=options[2],
        option4=options[3]
    )
    return label_answer,question


def preprocess(file_path, local_dir, data_source, split):
    dataset = read_jsonl(file_path)

    # base_instruction_following = "Let's think step by step. First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e., <think> reasoning process here </think>\n. If the final answer is obtained, use \\boxed{{}} to represent it."
    
    # box_instruction_following = "Let's think step by step. First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e., <think> reasoning process here </think>\n. Use \\possibleAnswer{} to represent all possible answers considered during the thinking process, and aim to include as many options as possible. If the final answer is obtained, use \\boxed{{}} to represent it."

    few_shot_instruction_following = """Try to solve the following question step by step. Please show your reasoning chain according to the following rules:
    1. First thinks about the reasoning chain in the mind and then provides the user with the answer. The reasoning chain is enclosed within <think> </think> tags, i.e., <think> reasoning chain here </think>\n.
    2. Label all segments that are potentially final results in the reasoning chain with \\possibleAnswer{{}} format. DO NOT label all the possible intermediate results, ONLY label the ones that could be the final answers, no matter it's correct or wrong. Label as many as you could.
    3. An example of the \\possibleAnswer{{}} annotation: "Wait, 5 times 360 is 1800, and 1800 divided by 36. Let's do that division: 1800 ÷ 36. Hmm, 36 times 50 is 1800, right? Because 36 x 50 is 1800. So, 1800 ÷ 36 = \\possibleAnswer{{50}}. Therefore, the degrees for cherry pie would be \\possibleAnswer{{50}} degrees."
    4. Label all segments that indicate a shift in reasoning within the text reasoning chain using the \\thoughtchange{{}} format. Label as many as you could.
    5.  An example of the \\thoughtchange{{}} annotation: "\\thoughtchange{{Wait, maybe}} I messed up the dailyprogress.\n\n\\thoughtchange{{Wait, hold on}}. If the original totaltime is T days, then when they switch to the newequipment after 1/3 of the tunnel is done, whichtook T/3 days, and then the remaining 2/3 is doneat a slower daily rate"  
    ### Question:{question}
    <｜Assistant｜><think>\n"""

    instruct_dataset = []

    for data in dataset:
        try:
            if data_source == "math":
                question = data["problem"] 
                answer = remove_boxed(last_boxed_only_string(data['solution']))  
            elif data_source == "gpqa":
                answer,question = make_gpqa_quesiton(data)
            else:
                question = data["Question"]
                answer = data['Answer']
        except:
            print(data_source)
            # print(data)


        question = few_shot_instruction_following.format(question=question)

            
        messages = [{"role": "user", "content": question}]
        d = {
            "messages": messages,
            "solution": answer
        }
        instruct_dataset.append(d)
    
    return instruct_dataset
    # save_path = os.path.join(local_dir, data_source + "_" + split + ".json")
    # with open(save_path,'w',encoding='utf-8') as f:
    #     json.dump(instruct_dataset, f, indent=4, ensure_ascii=False)

    

# preprocess("D:/BIT/NLP/datasets/OpenR1-Math-220k-2k.jsonl",local_dir="D:/BIT/NLP/并行/grpo_train/data/",data_source="openr1_2k",split="grpo_train")

# preprocess(file_path,local_dir,data_source=data_ls[idx],split="test")

train_ls = ["grpo_train/data/gpqa_train_400.jsonl","grpo_train/data/aime_train_400.jsonl","grpo_train/data/math_train_400.jsonl","grpo_train/data/TheoremQA_train_400.jsonl"]    

test_ls = ["grpo_train/data/gpqa_test_100.jsonl","grpo_train/data/aime_test_100.jsonl","grpo_train/data/math_test_100.jsonl","grpo_train/data/TheoremQA_test_100.jsonl"]  

data_ls = ['gpqa', 'aime', 'math', 'TheoremQA']

local_dir = "grpo_train/box_instruct/"

result = []
for idx, file_path in enumerate(train_ls):
    dataset = preprocess(file_path,local_dir,data_source=data_ls[idx],split="train")
    result = result + dataset

save_path = os.path.join(local_dir, "grpo_train.json")
with open(save_path,'w',encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
print("saved: ",save_path)

result = []
for idx, file_path in enumerate(test_ls):
    dataset = preprocess(file_path,local_dir,data_source=data_ls[idx],split="test")
    result = result + dataset

save_path = os.path.join(local_dir, "grpo_test.json")
with open(save_path,'w',encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
print("saved: ",save_path)