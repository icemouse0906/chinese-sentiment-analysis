import argparse
import pandas as pd
import random
import re
from tqdm import tqdm

# 简单同义词表、表情归一、错别字扰动
SYNONYMS = {'快': ['迅速', '飞快'], '好': ['棒', '优秀'], '慢': ['迟缓', '拖沓']}
EMOJI_MAP = {r'[😀😃😄😁]': '笑', r'[😢😭]': '哭', r'[👍]': '赞'}
TYPO_MAP = {'的': ['地', '得'], '了': ['啦', '喽']}


def synonym_replace(text, ratio=0.1):
    words = list(text)
    for i, w in enumerate(words):
        if w in SYNONYMS and random.random() < ratio:
            words[i] = random.choice(SYNONYMS[w])
    return ''.join(words)

def emoji_normalize(text):
    for pat, rep in EMOJI_MAP.items():
        text = re.sub(pat, rep, text)
    return text

def typo_perturb(text, ratio=0.05):
    words = list(text)
    for i, w in enumerate(words):
        if w in TYPO_MAP and random.random() < ratio:
            words[i] = random.choice(TYPO_MAP[w])
    return ''.join(words)

def main():
    parser = argparse.ArgumentParser(description='中文数据增强')
    parser.add_argument('--input', type=str, required=True, help='输入CSV，需含text列')
    parser.add_argument('--output', type=str, required=True, help='输出CSV')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--syn_ratio', type=float, default=0.1)
    parser.add_argument('--typo_ratio', type=float, default=0.05)
    args = parser.parse_args()
    random.seed(args.seed)
    df = pd.read_csv(args.input)
    texts = []
    for t in tqdm(df['text']):
        t1 = synonym_replace(t, args.syn_ratio)
        t2 = emoji_normalize(t1)
        t3 = typo_perturb(t2, args.typo_ratio)
        texts.append(t3)
    df['aug_text'] = texts
    df.to_csv(args.output, index=False)
    print(f'增强数据已保存到 {args.output}')

if __name__ == '__main__':
    main()
