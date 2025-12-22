import re

def process_string(text: str) -> str:
    """
    规则：
    1. 任何完整句子或单词连续重复 ≥3 次 → 整段删除（一次不留）。
    2. 无此类重复 → 从前 30 % 处切分，保留后半部分（完整句子边界）。
    """

    # 1. 删除连续重复 ≥3 次的完整句子
    text = re.sub(r'([^.!?]*[^.!?][.!?])(?:\s+\1){2,}', '', text)

    # 2. 删除连续重复 ≥3 次的完整单词
    def drop_word_triple(s: str) -> str:
        parts = s.split()
        out, i = [], 0
        while i < len(parts):
            token = parts[i]
            j = i
            while j < len(parts) and parts[j] == token:
                j += 1
            if j - i >= 3:          # 出现 ≥3 次
                i = j               # 整段跳过
            else:
                out.extend(parts[i:j])
                i = j
        return ' '.join(out)
    text = drop_word_triple(text)

    # 3. 若无剩余内容，直接返回空串
    if not text.strip():
        return ""

    # 4. 无重复时：从 30 % 处切分，保留后半句
    split_pos = int(len(text) * 0.3)
    match = re.search(r'[.!?]\s*', text[split_pos:])
    if match:
        boundary = split_pos + match.end()
        return text[boundary:].lstrip()
    else:
        return text[split_pos:].lstrip()



# 测试函数
if __name__ == "__main__":
    # 测试用例1：有连续重复子串
    test1 = "so the total calories in the bag would be 250 * 5 = 1250 calories. 250 * 5 = 1250 calories. 250 * 5 = 1250 calories. 250 * 5 = 1250 calories. 250 * 5 = 1250 calories. \\thought \\thought \\thought \\thought \\thought \\thought \\thought \\thought \\thought \\thought"
    print("测试1（有重复）：")
    print("原字符串：", test1)
    print("处理后：", process_string(test1))
    print()
    
    # 测试用例2：没有连续重复子串
    test2 = "Okay, let's see. The problem is about figuring out how many grams of chips someone can eat without exceeding their daily calorie target. The chips have a certain number of calories per serving, and the bag's total weight and servings are given. They've already consumed some calories, so we need to find out the remaining allowed grams. First, let me parse the information given. The bag of chips has 250 calories per serving. The entire bag is 300 grams and contains 5 servings. So, if the whole bag is 300g and 5 servings, each serving must be 300g divided by 5, which is 60 grams per serving. That makes sense because 5 servings times 60 grams each equals 300 grams. Now, the person's daily calorie target is 2000 calories, and they've already consumed 1800 calories. "
    print("测试2（无重复）：")
    print("原字符串：", test2)
    print("处理后：", process_string(test2))
    print()
    
