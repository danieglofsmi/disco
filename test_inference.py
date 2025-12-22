from transformers import AutoTokenizer
import json
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json
import re
import pandas as pd
import argparse

max_tokens = 16384
model_path = "/home/xyliang/model/share/DeepSeek-R1-Distill-Qwen-7B/"
name = "base"

# parser = argparse.ArgumentParser(description="Test script with command line arguments.")
# parser.add_argument("--lora_path", type=str, help="lora path")
# parser.add_argument("--model_name", type=str, help="model_name")
# parser.add_argument("--data_path", type=str, help="data_path")
# parser.add_argument("--dataset", type=str, help="dataset name: math,aime,openr1,TheoremQA,gsm8k")

# args = parser.parse_args()
# lora_path = args.lora_path
# name= args.model_name
# data_path = args.data_path
# dataset= args.dataset


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


sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.7, top_p=0.95,min_p=0,repetition_penalty=1.1)  

model = LLM(model=model_path,
            tensor_parallel_size=1, 
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len = max_tokens
           )
tokenizer = AutoTokenizer.from_pretrained(model_path)

def read_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)  # 解析每一行
            data.append(record)
    return data

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
    


def generate_all(input_file, model_name, dataset,max_data,output_path_base):
    prompts = []

    question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
    prompt = template_prompt.format(question=question)    
    prompts.append(prompt)

    # print("prompts[0]: ",prompts[0])

    outputs = model.generate(prompts, sampling_params)

    idx = 0
    for i, output in enumerate(outputs):
        idx += 1
        response = output.outputs[0].text.strip()

        question = data[i]["question"]
        label_answer = data[i]["answer"]

        ans = last_boxed_only_string(response)
        if ans:
            ans = remove_boxed_or_possibleAnswer(ans)
            correct = is_correct(ans,label_answer)
        else:
            correct = False        
        
        
        print({
            "id": str(idx),
            "correct": correct,
            "question": question,
            "label_answer": label_answer,
            "ans":ans,
            "response": response,
        })

    # print(f"Name: {name}")
    # print(f"dataset: {dataset}")
    # print(f"correct num: {correct_cnt}")


