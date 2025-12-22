import json
import random
import argparse
from collections import defaultdict
from math import sqrt
import numpy as np
from scipy.stats import chi2


def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["source"] == "amc2023": 
                key = (obj["source"], obj["question"])
                data[key] = obj["correct"]
    return data


# McNemar test (exact mid-p)
def mcnemar_test(b01, b10):
    """
    b01: baseline 错, method 对
    b10: baseline 对, method 错
    使用 McNemar mid-p 精确检验
    """
    n = b01 + b10
    if n == 0:
        return 1.0   # 完全一致，无显著性
    
    # Binomial tail probability (mid-p)
    from math import comb

    def binom_p(k):
        return comb(n, k) * (0.5 ** n)

    p_exact = sum(binom_p(k) for k in range(b01, n + 1))
    p_mid = p_exact - 0.5 * binom_p(b01)
    return p_mid



# Paired bootstrap t-test
def bootstrap_t_test(diffs, R=20000, seed=42):
    """
    diffs: method_correct - baseline_correct (paired differences)
    返回：
        mean_diff: 提升均值
        p_one_sided: method > baseline 的 p
        p_two_sided: method ≠ baseline 的 p
        ci: 95% percentile CI
    """
    random.seed(seed)
    n = len(diffs)
    diffs = np.array(diffs)
    observed_mean = np.mean(diffs)

    boot_means = np.random.choice(diffs, size=(R, n), replace=True).mean(axis=1)

    # one-sided (提升 > 0)
    p_one_sided = np.mean(boot_means <= 0)

    # two-sided (不论提升或下降，只要偏离 0)
    p_two_sided = np.mean(np.abs(boot_means) >= abs(observed_mean))

    # 95% CI
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)

    return observed_mean, p_one_sided, p_two_sided, (ci_lower, ci_upper)


def main(baseline_path, method_path):

    baseline = load_jsonl(baseline_path)
    method = load_jsonl(method_path)
    print(f"Loaded baseline: {len(baseline)} samples")
    print(f"Loaded method: {len(method)} samples")

    # 对齐
    keys = sorted(set(baseline.keys()) & set(method.keys()))
    # print(f"Aligned samples: {len(keys)}")
 
    ###########
    # baseline_keys = set(baseline.keys())
    # method_keys = set(method.keys())

    # only_in_baseline = baseline_keys - method_keys
    # only_in_method = method_keys - baseline_keys

    # print("\n=== Keys only in baseline ===")
    # # for k in sorted(only_in_baseline):
    # #     if(k[0] == 'aime24'):
    # #         print(k)

    # print("\n=== Keys only in method ===")
    # for k in sorted(only_in_method):
    #     print(k)

    # print(f"\nCount only in baseline: {len(only_in_baseline)}")
    # print(f"Count only in method: {len(only_in_method)}")
    ############


    baseline_correct = np.array([baseline[k] for k in keys], dtype=int)
    method_correct = np.array([method[k] for k in keys], dtype=int)

    n = len(keys)

    # 计算准确率
    acc_b = baseline_correct.mean()
    acc_m = method_correct.mean()
    print(f"Baseline accuracy: {acc_b:.4f}")
    print(f"Method   accuracy: {acc_m:.4f}")
    print(f"Absolute improvement: {acc_m - acc_b:.4f}")

    # ---------------------------------------------------
    # Bootstrap paired t-test
    # ---------------------------------------------------
    diffs = method_correct - baseline_correct
    mean_diff, p_one_sided, p_two_sided, ci = bootstrap_t_test(diffs)
    print("\n=== Bootstrap Paired t-test ===")
    print(f"Mean difference: {mean_diff:.4f}")
    print(f"p-value (one-sided): {p_one_sided:.6f}")
    # print(f"p-value (two-sided): {p_two_sided:.6f}")
    print(f"95% CI of improvement: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # ---------------------------------------------------
    # McNemar test
    # ---------------------------------------------------
    b01 = int(((baseline_correct == 0) & (method_correct == 1)).sum())
    b10 = int(((baseline_correct == 1) & (method_correct == 0)).sum())

    p_mcnemar = mcnemar_test(b01, b10)

    print("\n=== McNemar Test ===")
    print(f"b01 (baseline wrong, method correct): {b01}")
    print(f"b10 (baseline correct, method wrong): {b10}")
    print(f"McNemar exact mid-p value: {p_mcnemar:.6f}")



if __name__ == "__main__":
    baseline_file = "32b/32-base_test.jsonl"
    method_file = "32b/32b-diversity-grpo-step40_test.jsonl"
    # method_file = "7b-initial-new/label-7b-grpo-initial-step50-hf_amc2023.jsonl"
    

    main(baseline_file, method_file)
