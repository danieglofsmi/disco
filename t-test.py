import json
import numpy as np
from collections import defaultdict


def load_jsonl_to_dict(path):
    """
    读取 jsonl 文件，按 (source, question) 唯一键存储对应的 correct (0/1)。
    返回 dict: key=(source, question), value=0/1
    """
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            key = (obj["source"], obj["question"])
            data[key] = 1 if obj["correct"] is True else 0
    return data


def paired_bootstrap_t_test(baseline, method, B=10000, alternative='two-sided', seed=0):
    """
    baseline, method: 0/1 numpy arrays of same length n
    """
    rng = np.random.default_rng(seed)
    baseline = np.asarray(baseline, dtype=float)
    method = np.asarray(method, dtype=float)

    if baseline.shape != method.shape:
        raise ValueError("baseline and method lengths differ!")

    n = baseline.size
    d = method - baseline    # paired diff ∈ {-1, 0, 1}

    obs_mean = d.mean()
    sd = d.std(ddof=1)
    obs_se = sd / np.sqrt(n) if sd > 0 else 0
    obs_t = obs_mean / obs_se if obs_se > 0 else 0

    # Bootstrap
    boot_t = np.empty(B)
    boot_mean = np.empty(B)
    idx = np.arange(n)

    for i in range(B):
        samp = rng.choice(idx, size=n, replace=True)
        d_star = d[samp]
        mean_star = d_star.mean()
        sd_star = d_star.std(ddof=1)
        se_star = sd_star / np.sqrt(n) if sd_star > 0 else 0

        boot_mean[i] = mean_star
        boot_t[i] = mean_star / se_star if se_star > 0 else 0

    # Compute p-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(boot_t) >= abs(obs_t))
    elif alternative == 'greater':     # method > baseline
        p_value = np.mean(boot_t >= obs_t)
    elif alternative == 'less':        # method < baseline
        p_value = np.mean(boot_t <= obs_t)
    else:
        raise ValueError("alternative must be two-sided | greater | less")

    lower, upper = np.percentile(boot_mean, [2.5, 97.5])
    ci = (lower, upper)

    return {
        "n": n,
        "baseline_acc": baseline.mean(),
        "method_acc": method.mean(),
        "mean_diff": obs_mean,
        "observed_t": obs_t,
        "p_value": p_value,
        "ci_95_percentile": ci,
    }


def run_bootstrap_test(baseline_path, method_path,
                       B=10000, alternative="greater", seed=0):
    # --- load ---
    base = load_jsonl_to_dict(baseline_path)
    meth = load_jsonl_to_dict(method_path)

    # --- align by (source, question) ---
    keys = sorted(set(base.keys()) & set(meth.keys()))
    if len(keys) == 0:
        raise ValueError("No overlapping (source, question) pairs!")

    baseline_correct = np.array([base[k] for k in keys])
    method_correct   = np.array([meth[k] for k in keys])

    # --- run test ---
    result = paired_bootstrap_t_test(
        baseline_correct,
        method_correct,
        B=B,
        alternative=alternative,
        seed=seed
    )
    return result


if __name__ == "__main__":
    # 示例：替换为你自己的 jsonl 文件路径
    baseline_file = "32b/32-base_test.jsonl"
    method_file = "32b/32b-diversity-grpo-step40_test.jsonl"

    result = run_bootstrap_test(
        baseline_file,
        method_file,
        B=10000,            # 推荐 5k–10k
        alternative="greater",  # 单边检验：method 是否显著更好
        seed=42
    )

    print("===== Bootstrap Paired t-test Results =====")
    print(f"n = {result['n']}")
    print(f"Baseline accuracy = {result['baseline_acc']:.4f}")
    print(f"Method accuracy   = {result['method_acc']:.4f}")
    print(f"Mean difference (method - baseline) = {result['mean_diff']:.4f}")
    print(f"Observed t = {result['observed_t']:.4f}")
    print(f"95% percentile CI for diff = {result['ci_95_percentile']}")
    print(f"p-value = {result['p_value']:.6f}")
    print("==========================================")
