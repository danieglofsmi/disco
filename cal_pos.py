import json
from collections import defaultdict
import matplotlib.pyplot as plt
from verify import extract_answer_from_possibleAnswer, is_correct
import numpy as np

def find_pos(solution, ground_truth):
    original_solution = solution
    L = len(original_solution)

    all_results = []
    all_positions = []

    while solution:
        idx = solution.rfind("\\possibleAnswer{")
        if idx == -1:
            break

        abs_pos = len(original_solution) - len(solution) + idx
        all_positions.append(abs_pos)

        content = extract_answer_from_possibleAnswer(solution)
        if content is not None:
            all_results.append(content)

        solution = solution[:idx]

    all_results = all_results[::-1]
    all_positions = all_positions[::-1]

    correct_positions = []
    correct_cnt = 0

    for content, pos in zip(all_results, all_positions):
        try:
            if is_correct(content, ground_truth):
                correct_cnt += 1
                correct_positions.append(pos / L)
        except Exception as e:
            print(e)

    first_correct_pos = correct_positions[0] if correct_positions else None

    return correct_cnt, first_correct_pos, correct_positions


# ----------------------------------------------------
# 统计 jsonl 中所有数据的正确答案分布，包括 first_correct_pos
# ----------------------------------------------------
def collect_position_distributions(jsonl_path):
    source_positions = defaultdict(list)
    source_first_positions = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            solution = item["labeled_solution"]
            ground_truth = item["answer"]
            source = item["source"]

            if item['correct']:
                _, first_pos, correct_positions = find_pos(solution, ground_truth)

                # 记录所有正确答案位置
                source_positions[source].extend(correct_positions)

                # 记录 first_correct_pos
                if first_pos is not None:
                    source_first_positions[source].append(first_pos)

    return source_positions, source_first_positions





def plot_source_distributions(source_positions):
    for source, positions in source_positions.items():
        if not positions:
            print(f"[{source}] No correct answers.")
            continue

        plt.figure(figsize=(6,4))
        plt.hist(positions, bins=20)  # 默认直方图
        plt.xlabel("Position ratio (0~1)")
        plt.ylabel("Count")
        plt.title(f"Correct Answer Position Distribution: {source}")
        plt.tight_layout()
        plt.show()


def plot_first_correct_distributions(source_first_positions):
    for source, positions in source_first_positions.items():
        if not positions:
            print(f"[{source}] No first correct answers.")
            continue
        
        print(f"average first correct position for {source}: {sum(positions)/len(positions):.4f}")
        plt.figure(figsize=(6,4))
        plt.hist(positions, bins=20, range=(0,1))
        plt.xlabel("First correct answer position ratio (0~1)")
        plt.ylabel("Count")
        plt.title(f"Distribution of First Correct Answer Positions: {source}")
        plt.tight_layout()
        plt.show()


def analyze_distribution(positions):
    """
    positions: list of floats in [0,1]
    return: dict with mean, median, mode_bin, bin_counts
    """

    if not positions:
        return None

    positions = np.array(positions)

    mean_val = float(np.mean(positions))
    median_val = float(np.median(positions))

    # 分成 10 个等宽区间
    bin_counts, bin_edges = np.histogram(positions, bins=10, range=(0,1))
    mode_bin_index = int(np.argmax(bin_counts))

    return {
        "mean": mean_val,
        "median": median_val,
        "bin_counts": bin_counts.tolist(),
        "mode_bin": (float(bin_edges[mode_bin_index]), float(bin_edges[mode_bin_index+1]))
    }


def extract_first_possible_answer(solution):
    """Return the FIRST possibleAnswer{...} content, or None if missing."""
    idx = solution.find("\\possibleAnswer{")
    if idx == -1:
        return None

    sub = solution[idx:]
    return extract_answer_from_possibleAnswer(sub)


def first_answer_wrong_rate(jsonl_path):
    """
    Return:
        dict {source: {"total_correct": X, "wrong_first": Y, "wrong_rate": Y/X}}
    """
    total_correct = defaultdict(int)
    wrong_first = defaultdict(int)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            # 只统计模型最终回答正确的样本
            if not item.get("correct", False):
                continue

            solution = item["labeled_solution"]
            ground_truth = item["answer"]
            source = item["source"]

            total_correct[source] += 1

            # 获取第一个 possibleAnswer
            first_answer = extract_first_possible_answer(solution)
            if first_answer is None:
                # 没有 possibleAnswer，可按需视为错误 or 跳过，这里跳过
                continue

            try:
                if not is_correct(first_answer, ground_truth):
                    wrong_first[source] += 1
            except Exception as e:
                print("Error:", e)

    # —— 计算比例 —— #
    results = {}
    for src in total_correct:
        if total_correct[src] == 0:
            rate = None
        else:
            rate = wrong_first[src] / total_correct[src]

        results[src] = {
            "total_correct": total_correct[src],
            "wrong_first": wrong_first[src],
            "wrong_rate": rate,
        }

    return results


jsonl_path = "7b-grpo-base/label-7b-grpo-base_gsm8k.jsonl"
# jsonl_path = "7b-base/label-distill-7b-gsm8k-100.jsonl"
# jsonl_path = "7b-initial-new/label-7b-grpo-initial-step50-hf_aime24-25.jsonl"
source_positions, source_first_positions = collect_position_distributions(jsonl_path)
# plot_source_distributions(source_positions)
# plot_first_correct_distributions(source_first_positions)

for source, positions in source_first_positions.items():
    analysis = analyze_distribution(positions)
    print(source,analysis)

stats = first_answer_wrong_rate(jsonl_path)
for src, info in stats.items():
    print(src, info)

print("disco")
jsonl_path = "7b-initial-new/label-7b-grpo-initial-step50-hf_gsm8k.jsonl"
source_positions, source_first_positions = collect_position_distributions(jsonl_path)
# plot_source_distributions(source_positions)
# plot_first_correct_distributions(source_first_positions)

for source, positions in source_first_positions.items():
    analysis = analyze_distribution(positions)
    print(source,analysis)

stats = first_answer_wrong_rate(jsonl_path)
for src, info in stats.items():
    print(src, info)