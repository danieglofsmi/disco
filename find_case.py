import json

def compare_jsonl_files(file1_path, file2_path, output_path):
    """比较两个jsonl文件同一位置的记录，将correct不同的记录写入新文件"""
    diff_records = []

    with open(file1_path, 'r', encoding='utf-8') as f1, \
         open(file2_path, 'r', encoding='utf-8') as f2:
        
        for line_num, (line1, line2) in enumerate(zip(f1, f2), 1):
            try:
                record1 = json.loads(line1)
                record2 = json.loads(line2)
                
                if record1.get("correct") != record2.get("correct"):
                    diff_records.append({
                        "line_number": line_num,
                        "file1_record": record1,
                        "file2_record": record2
                    })
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析错误: {e}")
                continue

    # 写入差异到新jsonl文件
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for record in diff_records:
            out_file.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"比较完成，共找到{len(diff_records)}条差异记录，已写入: {output_path}")
    return diff_records

# 使用示例
if __name__ == "__main__":
    file1 = "7b-initial-new/label-7b-grpo-initial-step50-hf_gsm8k.jsonl"
    file2 = "7b-grpo-base/label-7b-grpo-base_gsm8k.jsonl"
    output = "differences.jsonl"
    
    compare_jsonl_files(file1, file2, output)