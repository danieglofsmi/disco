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

base_instruction_following = "Let's think step by step. First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e., <think> reasoning process here </think>\n. If the final answer is obtained, use \\boxed{{}} to represent it. \n### Question:{question}"

# instruction_following = """<｜User｜>Try to solve the following question step by step. Please show your reasoning chain according to the following rules:
# 1. First thinks about the reasoning chain in the mind and then provides the user with the answer. The reasoning chain is enclosed within <think> </think> tags, i.e., <think> reasoning chain here </think>\n.
# 2. Label all segments that are potentially final results in the reasoning chain with \\possibleAnswer{{}} format. DO NOT label all the possible intermediate results, ONLY label the ones that could be the final answers, no matter it's correct or wrong. Label as many as you could.
# 3. An example of the \\possibleAnswer{{}} annotation: "Wait, 5 times 360 is 1800, and 1800 divided by 36. Let's do that division: 1800 ÷ 36. Hmm, 36 times 50 is 1800, right? Because 36 x 50 is 1800. So, 1800 ÷ 36 = \\possibleAnswer{{50}}. Therefore, the degrees for cherry pie would be \\possibleAnswer{{50}} degrees."
# 4. Label all segments that indicate a shift in reasoning within the text reasoning chain using the \\thoughtchange{{}} format. Label as many as you could.
# 5.  An example of the \\thoughtchange{{}} annotation: "\\thoughtchange{{Wait, maybe}} I messed up the dailyprogress.\n\n\\thoughtchange{{Wait, hold on}}. If the original totaltime is T days, then when they switch to the newequipment after 1/3 of the tunnel is done, whichtook T/3 days, and then the remaining 2/3 is doneat a slower daily rate"  
# ### Question:{question}
# <｜Assistant｜><think>\n"""

def preprocess(file_path, data_source, split):
    dataset = read_jsonl(file_path)
    instruct_dataset = []
    idx = 1


    for data in dataset:
        if float(data["solved_percentage"])>80 or float(data["solved_percentage"])<20:
            continue
        try:
            question = data["problem"]
            answer = data['ground_truth']
        except:
            print(data_source)
            # print(data)


        data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "stem",
                "reward_model": {"style": "rule", "ground_truth": answer, "style":"rule"},
                "extra_info": {"split": split, "index": str(idx)},
            }    
        
        instruct_dataset.append(data)
        idx+=1
    
    return instruct_dataset


dataset = preprocess("deepscaler--difficulty.jsonl",data_source="DeepScaler",split="train")
save_path = os.path.join("grpo_train/verl/", "DeepScaler_grpo_train_questions.json")


df = pd.DataFrame(dataset)
train_df = df.sample(n=8000, random_state=42)
train_set = train_df.to_dict(orient='records')
with open(save_path,'w',encoding='utf-8') as f:
    json.dump(train_set, f, indent=4, ensure_ascii=False)

# df.to_json(save_path, orient="records", lines=False, force_ascii=False)
# train_df.to_parquet(os.path.join(save_path), engine="pyarrow")

