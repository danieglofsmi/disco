import json
import re
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple

def read_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)  # 解析每一行
            if not record['correct']:
                data.append(record)
    print(f"total: {len(data)}")
    return data

def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def box_cnt(solution):
    count = solution.count("\\possibleAnswer{")
    return count

def thought_cnt(solution):
    count = solution.count("\\thoughtchange{")
    return count

def alternatively_cnt(solution):
    count = solution.count("alternatively")+solution.count("Alternatively")
    return count

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
    
def all_boxed_contents(solution):
    contents = []
    
    while solution:
        if not "\\possibleAnswer{" in solution:
            break
        content = extract_answer_from_box(solution)
        if content is not None:
            contents.append(content)
    
        solution = solution[:solution.rfind("\\possibleAnswer{")]

    return contents

def unique_box_num(solution):
    unique_contents = set()
    
    while solution:
        if not "\\possibleAnswer{" in solution:
            break
        content = extract_answer_from_box(solution)
        if content is not None:
            unique_contents.add(content)
    
        solution = solution[:solution.rfind("\\possibleAnswer{")]

    return len(unique_contents)

def analysis(file_path):
    dataset = read_jsonl(file_path)
    counts_possibleAnswer_per_line = []
    counts_repeat_per_line = []
    counts_match_ans_per_line = []
    counts_match_answer_per_line = []
    counts_thought_per_line = []
    counts_alternatively_per_line = []
    counts_unique_box = []
    counts_correct_ratio = []

    cnt = 0
    
    for idx,data in enumerate(dataset):
        labeled_solution = data["labeled_solution"]
        ans_field = data["ans"]
        answer_field = data["answer"]
        correct = data['correct']

        # 1. 统计 \\possibleAnswer{ 出现次数
        box_num = box_cnt(labeled_solution)
        # if data['source'] == 'aime25' and box_num != 0:
        if box_num != 0:
            cnt += 1

            answers = all_boxed_contents(labeled_solution)
            counts_possibleAnswer_per_line.append(box_cnt(labeled_solution))
            counts_thought_per_line.append(thought_cnt(labeled_solution))
            counts_alternatively_per_line.append(alternatively_cnt(labeled_solution))
            counts_unique_box.append(unique_box_num(labeled_solution))

            # 2. 统计重复标签内容数量
            counter = Counter(answers)
            repeat_cnt = sum(v >= 2 for v in counter.values())
            counts_repeat_per_line.append(repeat_cnt)

            # 3. 统计与 ans 字段相同数量
            match_ans = sum(1 for a in answers if a.strip() == ans_field and not correct)
            counts_match_ans_per_line.append(match_ans)

            # 4. 统计与 answer 字段相同数量
            match_answer = sum(1 for a in answers if (a.strip() == ans_field and correct))
            counts_match_answer_per_line.append(match_answer)
            counts_correct_ratio.append(match_answer/len(answers))
    # f"#alternatively-avg={sum(counts_alternatively_per_line)/len(counts_alternatively_per_line):.2f}",
    print(  f"#thoughtchange-avg={sum(counts_thought_per_line)/len(counts_thought_per_line):.2f}, "
            f"#possibleAnswer-avg={sum(counts_possibleAnswer_per_line)/len(counts_possibleAnswer_per_line):.2f}, "
            f"#repeat-content-avg={sum(counts_repeat_per_line)/len(counts_repeat_per_line):.2f}, "
            f"#match_ans-false-avg={sum(counts_match_ans_per_line)/len(counts_match_ans_per_line):.2f}, "
            f"#match_answer-avg={sum(counts_match_answer_per_line)/len(counts_match_answer_per_line):.2f},"
            f"#unique-box-avg={sum(counts_unique_box)/len(counts_unique_box):.2f}, "
            f"#counts_correct_ratio={sum(counts_correct_ratio)/len(counts_correct_ratio):.5f}, "
            )
    print(cnt)

# # ---------------- 画图 ----------------
#     def plot_histogram(data: List[int], bins: int, title: str, xlabel: str, ylabel: str, filename: str) -> None:
#         plt.figure(figsize=(8, 5))
#         plt.hist(data, bins=bins, edgecolor='black', rwidth=0.8)
#         plt.title(title)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.tight_layout()
#         plt.savefig(filename)
#         plt.show()

#     # 自动确定合适的 bin 数：从 0 到 max_count+1
#     max_count = max(counts_possibleAnswer_per_line) if counts_possibleAnswer_per_line else 0
#     bins = range(max_count + 2)  # 例如 0,1,2,...,max+1

#     plot_histogram(counts_possibleAnswer_per_line,
#                    bins=bins,
#                    title="Distribution of possibleAnswer counts per line",
#                    xlabel="Number of \\possibleAnswer{}",
#                    ylabel="Frequency (number of lines)",
#                    filename="hist_possibleAnswer_counts.png")

    
# file_path = "7b-sft/label-7b-sft_amc2023.jsonl"
# analysis(file_path)
# file_path = "7b-grpo-base/label-7b-grpo-base_amc2023.jsonl"
dataset="gpqa_diamond"
print(dataset)
file_path = f"7b-grpo-base/label-7b-grpo-base_{dataset}.jsonl"
print("grpo base")
analysis(file_path)

# file_path = "7b-initial-new/label-7b-grpo-initial-step50-hf_amc2023.jsonl"
print("diversity")
file_path = f"7b-initial-new/label-7b-grpo-initial-step50-hf_{dataset}.jsonl"
analysis(file_path)