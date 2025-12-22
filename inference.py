import re
import uuid

import pandas as pd

from tqdm import tqdm
from joblib import Parallel,delayed
# from sklearn.metrics import classification_report
import requests
import os
import fcntl
import json
import time
from pathlib import Path

n_jobs = 8
#服务组名称
service_name = "7b-grpo-initial-step50-hf"
model_name = '7b-grpo-initial-step50-hf'
output_path_base = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/7b-initial-new/'

# model_name = "7b-initial-120"
# output_path_base = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/7b-initial-120-1/'

# 判断目录是否存在，不存在则创建
if not os.path.exists(output_path_base):
    os.makedirs(output_path_base, exist_ok=True)
    print(f"创建目录: {output_path_base}")

# template_prompt = """<｜User｜>Try to solve the following question step by step. Please show your reasoning chain according to the following rules:
#     1. First thinks about the reasoning chain in the mind and then provides the user with the answer. The reasoning chain is enclosed within <think> </think> tags, i.e., <think> reasoning chain here </think>\n.
#     2. Label all segments that are potentially final results in the reasoning chain with \\possibleAnswer{{}} format. DO NOT label all the possible intermediate results, ONLY label the ones that could be the final answers, no matter it's correct or wrong. Label as many as you could.
#     3. An example of the \\possibleAnswer{{}} annotation: "Wait, 5 times 360 is 1800, and 1800 divided by 36. Let's do that division: 1800 ÷ 36. Hmm, 36 times 50 is 1800, right? Because 36 x 50 is 1800. So, 1800 ÷ 36 = \\possibleAnswer{{50}}. Therefore, the degrees for cherry pie would be \\possibleAnswer{{50}} degrees."
#     4. Label all segments that indicate a shift in reasoning within the text reasoning chain using the \\thoughtchange{{}} format. Label as many as you could.
#     5.  An example of the \\thoughtchange{{}} annotation: "\\thoughtchange{{Wait, maybe}} I messed up the dailyprogress.\n\n\\thoughtchange{{Wait, hold on}}. If the original totaltime is T days, then when they switch to the newequipment after 1/3 of the tunnel is done, whichtook T/3 days, and then the remaining 2/3 is doneat a slower daily rate"  
#     ### Question:{question}
#     <｜Assistant｜><think>\n
#     """


template_prompt = """<｜User｜>Try to solve the following question step by step. If the final answer is obtained, use \\boxed{{}} to represent it.
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

def get_offline_server_res(messages,model_name,wsid):
    # ss_url = 'http://stream-server-online-openapi.turbotke.production.polaris:81/openapi/chat/completions'   #idc
    ss_url = "http://stream-server-online-openapi.turbotke.production.polaris:1081/openapi/chat/completions"  # devlcoud
    model = model_name  #服务组名称 spct_grpo
    wsid = wsid
   
    enable_stream = False

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer 7auGXNATFSKl7dF',
        'Wsid': wsid,
    }
    json_data = {
            "model": model,
            "query_id": 'test_query_id_' + str(uuid.uuid4()),
            "messages": messages,
            # "temperature": 0.2,
            # "top_p": 0.7,
            # "top_k": 150,
            "temperature": 0.0, ### !!!!!!!!!!!!!!!!!
            "top_p": 1,
            "top_k": 1,
            "repetition_penalty": 1.50,
            "output_seq_len": 15598,
            "max_input_seq_len": 4096,
            "stream": enable_stream,
        }
    # print("input: ",json_data)
    resp = requests.post(ss_url, headers=headers, json=json_data, stream=True)
    print('generated: ',resp.json())
    
    try:
        if resp.status_code == 200:
            resp_json = resp.json()
            return resp_json
            # # 处理可能缺少的键或None值
            # content = resp_json['choices'][0]["message"].get('content', '')
            # reasoning = resp_json['choices'][0]["message"].get('reasoning_content', '')
            # return (content + reasoning).strip()
        else:
            return ''
    except (ValueError, KeyError, IndexError) as e:
        print(resp_json)
        print(f"Error parsing response: {e}")
        return ''
    except Exception as e:
        print(resp_json)
        print(f"Unexpected error: {e}")
        return ''


def single_request_llm(data, offline_model_name, wsid, output_file='./output.jsonl'):
    try:
        input_prompt = data['messages']
        label_answer = data['answer']
        
        answers = []
        try_cnt = 5
        while try_cnt > 0:
            timeout_cnt = 5
            while timeout_cnt>0:
                try:
                    offline_res = get_offline_server_res(input_prompt, offline_model_name, wsid)
                    # print("get offline_res")
                    if not 'error' in offline_res.keys():
                        break
                except Exception as ex:
                    print(f"Error: {ex}")
                    timeout_cnt -= 1
                    time.sleep(3)
            #if offline_res != '':
            if not 'error' in offline_res.keys() and data['source']!='extra':
                resp_json = offline_res
                # print(offline_res)
                content = resp_json['choices'][0]["message"].get('content', '')
                reasoning = resp_json['choices'][0]["message"].get('reasoning_content', '')
                response = (reasoning + content).strip()
                # print(result)
                if response != "":
                    break
            else:
                try_cnt -= 1
                time.sleep(3)
                # print(offline_res)
        
        if not 'error' in offline_res.keys() and data['source']!='extra':
            ans = last_boxed_only_string(response)
            if ans:
                ans = remove_boxed_or_possibleAnswer(ans)
                correct = is_correct(ans,label_answer)
            else:
                correct = False        
            
            result = {
                "source": data['source'],
                "correct": correct,
                "ans":ans,
                "label_answer": data['answer'],
                "question": data['question'],
                "response": response,
            }
            
            # 互斥写入文件
            with open(output_file, 'a') as f:
                # 获取文件锁
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    # 写入JSON行数据
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                finally:
                    # 释放文件锁
                    fcntl.flock(f, fcntl.LOCK_UN)
        
        # return data
    except Exception as ex:
        print(f"Error in single_request_llm: {ex}")
        return {}    # return int(re.findall("oxed\{(.*?)\}",llm_output,re.DOTALL)[0])

def extract_answer_from_possibleAnswer(solution):
    start = solution.rfind("\\possibleAnswer{")
    if start == -1:
        return None

    start += len("\\possibleAnswer{")
    bracket_count = 1  
    end = start

    while end < len(solution) and bracket_count > 0:
        if solution[end] == '{':
            bracket_count += 1
        elif solution[end] == '}':
            bracket_count -= 1
        end += 1

    if bracket_count == 0:
        return solution[start:end - 1]  
    else:
        return None
    
def remove_boxed_or_possibleAnswer(s):
    def extract_content(s, start_marker, end_marker):
        # 找到最内层的 start_marker 和 end_marker
        start = s.find(start_marker)
        if start == -1:
            return None
        
        start += len(start_marker)
        bracket_count = 1
        end = start

        while end < len(s) and bracket_count > 0:
            if s[end] == '{':
                bracket_count += 1
            elif s[end] == '}':
                bracket_count -= 1
            end += 1

        if bracket_count == 0:
            return s[start:end - 1]
        else:
            return None

    # 递归提取内容
    def recursive_extract(s):
        while True:
            # 尝试提取 \\boxed{...}
            content = extract_content(s, "\\boxed{", "}")
            if content is not None:
                s = content
                continue

            # 尝试提取 \\possibleAnswer{...}
            content = extract_content(s, "\\possibleAnswer{", "}")
            if content is not None:
                s = content
                continue

            # 如果没有更多嵌套，返回最终结果
            return s

    # 调用递归提取函数
    return recursive_extract(s)


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\possibleAnswer")
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

def process_subnum(s):
    # 定义下标数字到普通数字的映射
    subscript_to_normal = {
        '₀': '0',
        '₁': '1',
        '₂': '2',
        '₃': '3',
        '₄': '4',
        '₅': '5',
        '₆': '6',
        '₇': '7',
        '₈': '8',
        '₉': '9'
    }

    # 替换下标数字，并在数字之间添加下划线
    result = []
    for char in s:
        if char in subscript_to_normal:
            result.append('_')  # 在下标数字前添加下划线
            result.append(subscript_to_normal[char])  # 替换为普通数字
        else:
            result.append(char)

    # 将结果拼接成字符串
    new_s = ''.join(result)
    return new_s

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def process_string(s):
    try:
        s = s.replace("\n", "")
        s = s.replace("\\\\", "\\") # replace \\ with \
        s = s.replace("^{\\circ}", "") # Remove circ (degrees)
        s = s.replace("^\\circ", "")
        s = s.replace("\\tfrac", "\\frac")  
        s = s.replace("\\dfrac", "\\frac")  
        s = s.replace("\\left(", "(")
        s = s.replace("\\right)", ")")
        s = s.replace("\\%", "") 
        s = s.replace(".00", "") 
        s = s.replace(" .", " 0.")
        s = s.replace("{.", "{0.")
        s = s.replace("\\$", "")
        s = s.replace("\!", "")
        s = s.replace("\\!", "")
        s = s.replace("\\ ", " ")
        s = re.sub(r"\s+", "", s) 

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(s.split("=")) == 2:
            if len(s.split("=")[0]) <= 2:
                s = s.split("=")[1]

        s = fix_sqrt(s)  # fix sqrt3 --> sqrt{3}
        s = fix_fracs(s) # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}
        s = fix_sqrt(s) # fix sqrt3 --> sqrt{3}
        s = fix_a_slash_b(s) # X/Y changed to \frac{X}{Y}

        if s.isdigit():
            s = process_subnum(s)
            s = int(s)
        elif s.replace('.', '', 1).isdigit() and s.count('.') <= 1:
            s = float(s)
        else:
            if s[0].isdigit():
                s = s.replace(",", "")   
            
            # if len(s)>2 and ((s[0] == '[' and s[-1] == ']') or (s[0] == '(' and s[-1] == ')')):
            #     s = s[1:-1]   

            if s and s[0] in ('A','B','C','D'):
                s = s[0]
            elif len(s) > 1:
                s = s.lower() 
        return s
    
    except:
        return str(s)
    

def is_correct(ans, label_answer):
    if not ans or ans is None:  # 明确检查 ans 是否为空或 None
        return False

    label_answer = process_string(label_answer)
    ans = process_string(ans)

    if isinstance(label_answer, int) or isinstance(label_answer, float):
        # 如果 label_answer 是数字类型，直接比较
        return ans == label_answer
    else:
        # 如果 label_answer 是字符串类型，检查子字符串关系
        return str(ans) in label_answer or label_answer in str(ans)
    


def generate_all(input_file, model_name, dataset,output_path_base):

    output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')

    _, extension = os.path.splitext(input_file)
    extension = extension.lower()

    if extension == '.json':
        dataset = read_json(input_file)
    elif extension == '.jsonl':
        dataset = read_jsonl(input_file)
    
    new_dataset = []
    for data in dataset:
        d = dict()
        d['source'] = data['source']
        d['question'] = data['question']
        d['messages']=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':template_prompt.format(question=data['question'])}]
        d['answer'] = data['answer']
        new_dataset.append(d)

    # df = pd.DataFrame(new_dataset)
    # print(df['messages'].iloc[0])
    # print(len(df))
    
    existed = set()
    if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    existed.add(json.loads(line)['question'])
    print(f'existed: {len(existed)}')
    # data_points = df.to_dict(orient='records')
    data_points = [x for x in new_dataset if x['question'] not in existed]
    print(len(data_points))

    Parallel(n_jobs=n_jobs)(delayed(single_request_llm)(data, service_name,'11331', output_file) for data in tqdm(data_points))
    

    # idx = 0
    # correct_cnt = 0
    # for i, response in enumerate(outputs):
    #     idx += 1

    #     question = data[i]["question"]
    #     label_answer = data[i]["answer"]

    #     ans = last_boxed_only_string(response)
    #     if ans:
    #         ans = remove_boxed_or_possibleAnswer(ans)
    #         correct = is_correct(ans,label_answer)
    #         correct_cnt += 1
    #     else:
    #         correct = False        
        
    #     result = {
    #         "id": str(idx),
    #         "correct": correct,
    #         "question": question,
    #         "label_answer": label_answer,
    #         "ans":ans,
    #         "response": response,
    #     }
        
    #     with open(output_file, 'a', encoding='utf-8') as f:
            # f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # print(f'model: {model_name}\ndataset: {dataset}\ncorrect cnt: {correct_cnt}')

try_cnt = 50
input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/testset/merge_test.jsonl'
dataset = "test"
output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
generate_all(input_file, model_name, dataset,output_path_base)
while try_cnt > 0:
    existed_data = read_jsonl(output_file)
    if len(existed_data) >= 2144:
        break
    generate_all(input_file, model_name, output_path_base)
    try_cnt -= 1
    time.sleep(5)
    print(f'test try_cnt: {try_cnt}')

# try_cnt = 80
# 


# try_cnt = 50
# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/testset_100/merge_test_100.jsonl'
# dataset = "test100"
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 530:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1
#     time.sleep(5)
#     print(f'test100 try_cnt: {try_cnt}')





# try_cnt = 20
# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/testset/amc2023.jsonl'
# dataset = 'amc2023'
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 40:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1
#     time.sleep(5)

# try_cnt = 20
# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/testset/aime_test.jsonl'
# dataset = 'aime'
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 90:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1
#     time.sleep(5)


# try_cnt = 20
# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/testset_100/math_test_100.jsonl'
# dataset = 'math'
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 100:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1
#     time.sleep(5)

# try_cnt = 20
# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/testset_100/gsm8k_test_100.jsonl'
# dataset = 'gsm8k'
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 100:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1
#     time.sleep(5)

# try_cnt = 20
# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/testset_100/gpqa_test_100.jsonl'
# dataset = 'gpqa'
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 100:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1

# try_cnt = 20
# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/testset_100/TheoremQA_test_100.jsonl'
# dataset = 'TheoremQA'
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 100:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1
#     time.sleep(5)

# try_cnt = 20
# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/combined_27.json'
# dataset = 'valid27'
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 27:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1

# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/Combined_grpo_valid_raw.json'
# dataset = 'valid100'
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 100:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1

# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/testset/math500.jsonl'
# dataset = 'math500'
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 500:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1

# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/testset/gpqa_diamond.jsonl'
# dataset = 'gpqa_diamond'
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 198:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1

# time.sleep(1)

# input_file = '/apdcephfs/cfs_cloud/users/brentbxu/diverisity-grpo/testset/gsm8k_test.jsonl'
# dataset = 'gsm8k_test'
# output_file = os.path.join(output_path_base, f'{model_name}_{dataset}.jsonl')
# generate_all(input_file, model_name, dataset,output_path_base)
# while try_cnt > 0:
#     existed_data = read_jsonl(output_file)
#     if len(existed_data) >= 1319:
#         break
#     generate_all(input_file, model_name, dataset,output_path_base)
#     try_cnt -= 1

# time.sleep(1)
