import re
import json

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


    def recursive_extract(s):
        while True:
            # extracting \\boxed{...}
            content = extract_content(s, "\\boxed{", "}")
            if content is not None:
                s = content
                continue

            # extracting \\possibleAnswer{...}
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
        # s = remove_right_units(s)

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

    label_answer = process_string(label_answer)
    ans = process_string(ans)

    if isinstance(label_answer, int) or isinstance(label_answer, float):
        return ans == label_answer
    else:
        return str(ans) in label_answer or label_answer in str(ans)