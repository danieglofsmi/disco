import re
import matplotlib.pyplot as plt

# 日志文件路径
log_file = "train.log"

# 正则表达式模式
patterns = {
    "global_step": r"- training/global_step:\s*([\d\.]+)",
    "critic_score_mean": r"- critic/score/mean:\s*([\d\.]+)",
    "actor_entropy_loss": r"- actor/entropy_loss:\s*([\d\.]+)",
    "response_length_mean": r"- response_length/mean:\s*([\d\.]+)"
}

# 初始化列表
global_steps = []
critic_scores = []
actor_entropy_losses = []
response_lengths = []

# 逐行读取日志文件
with open(log_file, "r", encoding="utf-8") as infile:
    for line in infile:
        # 检查是否以 "step:" 开头
        if line.strip().startswith("step:"):
            # 提取所需的数值
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    value = float(match.group(1))
                    if key == "global_step":
                        global_steps.append(value)
                    elif key == "critic_score_mean":
                        critic_scores.append(value)
                    elif key == "actor_entropy_loss":
                        actor_entropy_losses.append(value)
                    elif key == "response_length_mean":
                        response_lengths.append(value)

# 检查数据长度是否一致
if len(global_steps) != len(critic_scores) or len(global_steps) != len(actor_entropy_losses) or len(global_steps) != len(response_lengths):
    raise ValueError("数据长度不一致，请检查日志文件和正则表达式。")

# 绘制 critic_score_mean 关于 global_step 的折线图
plt.figure(figsize=(10, 6))
plt.plot(global_steps, critic_scores, label="Critic Score Mean")
plt.title("Critic Score Mean vs. Global Step")
plt.xlabel("Global Step")
plt.ylabel("Critic Score Mean")
plt.grid(True)
plt.legend()
plt.show()

# 绘制 actor_entropy_loss 关于 global_step 的折线图
plt.figure(figsize=(10, 6))
plt.plot(global_steps, actor_entropy_losses, label="Actor Entropy Loss")
plt.title("Actor Entropy Loss vs. Global Step")
plt.xlabel("Global Step")
plt.ylabel("Actor Entropy Loss")
plt.grid(True)
plt.legend()
plt.show()

# 绘制 response_length_mean 关于 global_step 的折线图
plt.figure(figsize=(10, 6))
plt.plot(global_steps, response_lengths, label="Response Length Mean")
plt.title("Response Length Mean vs. Global Step")
plt.xlabel("Global Step")
plt.ylabel("Response Length Mean")
plt.grid(True)
plt.legend()
plt.show()