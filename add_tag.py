#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给 JSONL 每条数据增加固定 source 字段，结果写入新文件。
"""

import json
from pathlib import Path

INPUT_FILE  = Path("7b-grpo-base/label-7b-grpo-base_gsm8k.jsonl")          # 原始文件
OUTPUT_FILE = Path("7b-grpo-base/label-7b-grpo-base_gsm8k_with_source.jsonl")  # 结果文件
SOURCE_VAL  = "gsm8k"              # 要加的 source 值

def add_source_field(input_file: Path, output_file: Path, source_val: str):
    print(input_file)
    with input_file.open("rt", encoding="utf-8") as fin, \
         output_file.open("wt", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"第 {line_no} 行 JSON 非法: {e}")
            obj["source"] = source_val
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    add_source_field(INPUT_FILE, OUTPUT_FILE, SOURCE_VAL)
    print(f"✅ 处理完成，结果已写入: {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()