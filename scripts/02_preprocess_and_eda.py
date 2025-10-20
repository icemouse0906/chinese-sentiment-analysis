# coding: utf-8
"""
预处理与快速 EDA 脚本
- 读取三个数据集（自动检测编码）
- 基本统计：样本数、文本长度、长度分布直方图（保存为 PNG）
- 分词（jieba），去停用词（可选自定义停用词文件），输出处理后 CSV 到 data/processed_*.csv

使用方法（在项目根目录）：
python scripts/02_preprocess_and_eda.py

生成：
- output/eda_stats.csv（各数据集基本统计）
- output/length_hist_*.png
- data/processed_*.csv

注意：需要安装 requirements.txt 中的依赖。
"""

from pathlib import Path
import chardet
import pandas as pd
import jieba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import codecs

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'NLP数据集'
OUT = ROOT / 'output'
DATA_OUT = ROOT / 'data'
OUT.mkdir(exist_ok=True)
DATA_OUT.mkdir(exist_ok=True)

FILES = {
    'hotel': DATA_DIR / '酒店评论数据' / 'ChnSentiCorp_htl_all.csv',
    'ecommerce': DATA_DIR / '电商评论数据' / 'online_shopping_10_cats.csv',
    'waimai': DATA_DIR / '外卖评论数据' / 'waimai_10k.csv'
}

STOPWORDS_PATH = ROOT / 'stopwords.txt'  # 可选，如果没有会用内置小列表

# 小型内置停用词备选
DEFAULT_STOPWORDS = set(['的','了','和','是','我','也','很','在','有','就','都','不','人','这'])


def detect_encoding(path, nbytes=10000):
    with open(path, 'rb') as f:
        raw = f.read(nbytes)
    return chardet.detect(raw)['encoding']


def read_csv_auto(path):
    enc = detect_encoding(path)
    print(f'Read {path.name} enc={enc}')
    # Try the detected encoding first, then fall back to a small list of safe encodings.
    # Use engine='python' and on_bad_lines='skip' to be resilient to malformed rows.
    tried = []
    for try_enc in [enc, 'utf-8', 'gb18030', 'latin1']:
        if not try_enc or try_enc in tried:
            continue
        tried.append(try_enc)
        try:
            df = pd.read_csv(path, encoding=try_enc, on_bad_lines='skip', engine='python')
            print(f'Success reading {path.name} with encoding={try_enc}')
            return df
        except Exception as e:
            print(f'Failed reading {path.name} with encoding={try_enc}: {e}')
    # Last resort: read without specifying encoding (will use system default)
    print(f'All encoding attempts failed for {path.name}, trying without explicit encoding')
    df = pd.read_csv(path, on_bad_lines='skip', engine='python')
    return df


def load_stopwords():
    if STOPWORDS_PATH.exists():
        return set([w.strip() for w in STOPWORDS_PATH.read_text(encoding='utf-8').splitlines() if w.strip()])
    return DEFAULT_STOPWORDS


def simple_clean(text):
    if pd.isna(text):
        return ''
    s = str(text).replace('\n', ' ').replace('\r', ' ').strip()
    # If the text contains literal escape sequences like '\u4e2d' or '\xE5', decode them
    if re.search(r'\\u[0-9a-fA-F]{4}', s) or re.search(r'\\x[0-9a-fA-F]{2}', s):
        try:
            # codecs.decode with 'unicode_escape' will turn '\uXXXX' into actual unicode
            s_dec = codecs.decode(s, 'unicode_escape')
            # Only accept the decoded version if it contains any non-ASCII characters (likely Chinese)
            if any(ord(ch) > 127 for ch in s_dec):
                s = s_dec
        except Exception:
            # if decoding fails, keep original
            pass
    return s


def tokenize(text, stopwords):
    segs = [w for w in jieba.cut(text) if w.strip() and w not in stopwords]
    return segs


if __name__ == '__main__':
    stats = []
    stopwords = load_stopwords()
    for name, path in FILES.items():
        if not path.exists():
            print('Missing', path)
            continue
        df = read_csv_auto(path)
        # Normalize: many files have header 'review' or 'cat,review'
        if 'review' in df.columns:
            text_col = 'review'
        elif 'content' in df.columns:
            text_col = 'content'
        else:
            # fallback to first column
            text_col = df.columns[0]
        df[text_col] = df[text_col].apply(simple_clean)
        df['char_len'] = df[text_col].str.len()
        df['word_count'] = df[text_col].str.split().apply(lambda x: len(x) if isinstance(x, list) else (len(str(x).split()) if pd.notna(x) else 0))

        n = len(df)
        avg_len = df['char_len'].mean()
        median_len = df['char_len'].median()
        stats.append({'file': path.name, 'nrows': n, 'avg_len': avg_len, 'median_len': median_len})

        # 保存长度分布图
        plt.figure(figsize=(6,4))
        plt.hist(df['char_len'].clip(upper=500), bins=50)
        plt.title(f'{name} char length distribution')
        plt.xlabel('chars')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(OUT / f'length_hist_{name}.png')
        plt.close()

        # 分词并写入处理后的 CSV（加入 tokens 列和 tokenized join）
        df['tokens'] = df[text_col].apply(lambda t: tokenize(t, stopwords))
        df['tokens_join'] = df['tokens'].apply(lambda toks: ' '.join(toks))
        out_path = DATA_OUT / f'processed_{name}.csv'
        # Force UTF-8 when writing processed CSVs so downstream consumers see proper Chinese text
        df.to_csv(out_path, index=False, encoding='utf-8')
        print(f'Wrote processed data to {out_path}')

    # Ensure stats are UTF-8 encoded as well
    pd.DataFrame(stats).to_csv(OUT / 'eda_stats.csv', index=False, encoding='utf-8')
    print('Wrote stats to', OUT / 'eda_stats.csv')
