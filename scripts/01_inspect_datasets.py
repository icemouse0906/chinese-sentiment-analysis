#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的数据检查脚本：
- 读取工作区中的三个 CSV 数据集
- 尝试检测编码并用 pandas 读取
- 打印行数、列名、前 5 行样本
- 将每个文件的前 20 行写入 output/samples_inspection.csv
"""
import os
import sys
from pathlib import Path
import chardet
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "NLP数据集"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

FILES = [
    DATA_DIR / "酒店评论数据" / "ChnSentiCorp_htl_all.csv",
    DATA_DIR / "电商评论数据" / "online_shopping_10_cats.csv",
    DATA_DIR / "外卖评论数据" / "waimai_10k.csv",
]


def detect_encoding(path, nbytes=10000):
    with open(path, 'rb') as f:
        raw = f.read(nbytes)
    return chardet.detect(raw)['encoding']


def try_read_csv(path):
    enc = None
    try:
        enc = detect_encoding(path)
    except Exception:
        enc = 'utf-8'
    print(f"Reading {path.name} detected-encoding={enc}")
    try:
        # first try with detected encoding
        df = pd.read_csv(path, encoding=enc)
    except Exception as e:
        print(f"Failed with encoding {enc}: {e}, trying utf-8 with on_bad_lines='skip'")
        try:
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip', engine='python')
        except Exception:
            # final fallback: latin1 (will not error on arbitrary bytes)
            print('Falling back to latin1 with on_bad_lines=skip')
            df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip', engine='python')
    return df


if __name__ == '__main__':
    rows = []
    for p in FILES:
        if not p.exists():
            print(f"Missing: {p}")
            continue
        df = try_read_csv(p)
        print("-"*40)
        print(p)
        print("shape:", df.shape)
        print("columns:", list(df.columns))
        print("head:\n", df.head(5).to_string())
        sample_out = OUT_DIR / f"samples_{p.stem}.csv"
        df.head(20).to_csv(sample_out, index=False)
        print(f"Wrote sample to {sample_out}")
        rows.append({'file': p.name, 'shape': df.shape, 'columns': '|'.join(df.columns.astype(str))})

    summary = pd.DataFrame(rows)
    summary.to_cdatasets_summarysv(OUT_DIR / '.csv', index=False)
    print("Summary written to output/datasets_summary.csv")
