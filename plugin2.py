import re
from typing import List

from swift.plugin import ORM, orms
from swift.utils import get_logger
from math_verify import parse, verify

logger = get_logger()

def calculate_Format(solution_str):
    format_reward = 0.0
    if "</think>" in solution_str:
        format_reward+=0.5
    if "//boxed" in solution_str:
        format_reward+=0.5
    return format_reward

def calculate_Accuracy(content,solution):
    ans = last_boxed_only_string(content)
    if ans:
        ans = remove_boxed_or_possibleAnswer(ans)
        score = float(is_correct(ans,solution))
    else:
        parse_label_answer = parse(str(solution))
        parse_ans = parse(content)
        if verify(parse_label_answer, parse_ans):
            score = 1.0
        else:
            score = 0.0
        
    return score


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
    
    else:
        label_answer = process_string(label_answer)
        ans = process_string(ans)

        if isinstance(label_answer, int) or isinstance(label_answer, float):
            # 如果 label_answer 是数字类型，直接比较
            return ans == label_answer
        else:
            # 如果 label_answer 是字符串类型，检查子字符串关系
            return str(ans) in label_answer or label_answer in str(ans)


    
def extract_answer_from_box(solution):
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
    
def box_cnt_val(solution, cnt_weight):
    count = solution.count("\\possibleAnswer{")
    return count*cnt_weight

def thought_cnt_val(solution, cnt_weight):
    count = solution.count("\\thoughtchange{")
    return count*cnt_weight

def box_diversity_val(solution, diversity_weight):
    unique_contents = set()
    
    while solution:
        if not "\\possibleAnswer{" in solution:
            break
        content = extract_answer_from_box(solution)
        if content is not None:
            unique_contents.add(content)
    
        solution = solution[:solution.rfind("\\possibleAnswer{")]

    return len(unique_contents)*diversity_weight

def box_acc_val(solution, ground_truth, correct_weight):
    
    all_results = []
    while solution:
        if not "\\possibleAnswer{" in solution:
            break
        content = extract_answer_from_possibleAnswer(solution)
        if content is not None:
            all_results.append(content)
    
        solution = solution[:solution.rfind("\\possibleAnswer{")]
    correct_cnt = 0
    for answer in all_results:
        try:
            if is_correct(answer, ground_truth):
                correct_cnt += 1
        except Exception as e:
            print(e)
            
    return correct_cnt*correct_weight



class Diversiy(ORM):
    #def __init__(self):
    #    from math_verify import LatexExtractionConfig, parse, verify
    #    from latex2sympy2_extended import NormalizationConfig

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        Final_diversity_weight = 1.0

        cnt_weight = 0.1
        diversity_weight = 0.2
        correct_weight = 0.3

        box_cnt_val_max=1.5
        thought_cnt_max = 2.0
        box_diversity_max = 3.0
        box_correct_max = 4.5

        for completion, ground_truth in zip(completions, solution):
            box_cnt_score = min(box_cnt_val(completion,cnt_weight),box_cnt_val_max)    # clip
            thought_cnt_score = min(thought_cnt_val(completion,cnt_weight),thought_cnt_max)
            box_diversity_score = min(box_diversity_val(completion,diversity_weight),box_diversity_max)
            box_correct_score = min(box_acc_val(completion, ground_truth, correct_weight),box_correct_max)
            # 1
            diversity_reward = box_cnt_score + thought_cnt_score + box_diversity_score + box_correct_score
            print(f"diversity_reward: {diversity_reward}, box_cnt_score: {box_cnt_score}, thought_cnt_score:{thought_cnt_score} ,box_diversity_score:{box_diversity_score} ,box_correct_score:{box_correct_score}")
            rewards.append(Final_diversity_weight*diversity_reward)
        return rewards


class MathAccuracy(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        Final_acc_weight = 2.0
        rewards = []
        for content, sol in zip(completions, solution):
            accuracy_reward = calculate_Accuracy(content, sol)
            rewards.append(Final_acc_weight*accuracy_reward)
        return rewards

class MathFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        # pattern = r'^<think>.*?</think>\n<answer>.*?</answer>(?![\s\S])'
        # matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        rewards = []
        Final_format_weight = 1.0
        for content in completions:
            format_reward = calculate_Format(Final_format_weight*content)
            rewards.append(format_reward)
        return rewards

orms['accuracy'] = MathAccuracy
orms['diversity'] = Diversiy
orms['format'] = MathFormat