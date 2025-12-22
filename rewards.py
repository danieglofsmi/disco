import re
from math_verify import parse, verify

def my_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    res = compute_score(solution_str, ground_truth)
    return res


def compute_score(solution_str, ground_truth) -> float:
    score = dict()
    retval = 0.
    try:
        Final_acc_weight = 2.0
        Final_diversity_weight = 1.0
        Final_format_weight = 1.0

        cnt_weight = 0.1
        diversity_weight = 0.2
        correct_weight = 0.3

        box_cnt_val_max=1.5
        thought_cnt_max = 2.0
        box_diversity_max = 3.0
        thought_diversity_max = 4.0
        box_correct_max = 4.5

       
        box_cnt_score = min(box_cnt_val(solution_str,cnt_weight),box_cnt_val_max)    # clip
        thought_cnt_score = min(thought_cnt_val(solution_str,cnt_weight),thought_cnt_max)
        thought_diversity_score = min(thought_diversity_val(solution_str,diversity_weight),thought_diversity_max)
        box_diversity_score = min(box_diversity_val(solution_str,diversity_weight),box_diversity_max)
        box_correct_score = min(box_acc_val(solution_str, ground_truth, correct_weight),box_correct_max)

        diversity_reward = box_cnt_score + thought_cnt_score + box_diversity_score + box_correct_score
        diversity_reward = box_cnt_score + box_diversity_score + box_correct_score
        diversity_reward = thought_cnt_score + thought_diversity_score
    
        accuracy_reward = calculate_Accuracy(solution_str, ground_truth)

        format_reward = calculate_Format(solution_str)

        # print(f"diversity_reward: {diversity_reward}, box_cnt_score: {box_cnt_score}, thought_cnt_score:{thought_cnt_score} ,box_diversity_score:{box_diversity_score} ,box_correct_score:{box_correct_score}")
        # print(f"accuracy_reward: {accuracy_reward}")

        retval = Final_diversity_weight * diversity_reward + Final_acc_weight * accuracy_reward + Final_format_weight*format_reward


    except Exception as e:
        print(e)
        retval = 0.0

    score["score"] = retval
    score["accuracy_score"] = accuracy_reward
    score["diversity_score"] = diversity_reward
    score["box_cnt_score"] = box_cnt_score
    score["thought_cnt_score"] = thought_cnt_score
    score["thought_diversity_score"] = thought_diversity_score
    score["box_diversity_score"] = box_diversity_score
    score["box_correct_score"] = box_correct_score
    score["format_score"] = format_reward

    return score

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
        # find the innermost layer start_marker 和 end_marker
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

    # Recursive content extraction
    def recursive_extract(s):
        while True:
            # extract \\boxed{...}
            content = extract_content(s, "\\boxed{", "}")
            if content is not None:
                s = content
                continue

            # extract \\possibleAnswer{...}
            content = extract_content(s, "\\possibleAnswer{", "}")
            if content is not None:
                s = content
                continue

            return s


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
    # mapping from subscript numbers to regular numbers
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

    # Replace the subscript numbers and add _ between the numbers.
    result = []
    for char in s:
        if char in subscript_to_normal:
            result.append('_')  
            result.append(subscript_to_normal[char]) 
        else:
            result.append(char)

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
    if not ans or ans is None:  
        return False
    
    else:
        label_answer = process_string(label_answer)
        ans = process_string(ans)

        if isinstance(label_answer, int) or isinstance(label_answer, float):
            return ans == label_answer
        else:
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

def extract_thought_from_box(solution):
    start = solution.rfind("\\thoughtchange{")
    if start == -1:
        return None

    start += len("\\thoughtchange{")
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

def thought_diversity_val(solution, diversity_weight):
    unique_contents = set()
    
    while solution:
        if not "\\thoughtchange{" in solution:
            break
        content = extract_thought_from_box(solution)
        if content is not None:
            unique_contents.add(content)
    
        solution = solution[:solution.rfind("\\thoughtchange{")]

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

