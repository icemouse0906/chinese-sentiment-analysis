import pandas as pd
import numpy as np
import argparse
import json
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description='主题-情感联合分析')
    parser.add_argument('--topic_file', type=str, required=True, help='LDA主题分配结果CSV，需含topic列')
    parser.add_argument('--sentiment_file', type=str, required=True, help='情感标签CSV，需含sentiment_label列')
    parser.add_argument('--output', type=str, default='output/topic_sentiment_matrix.csv', help='输出矩阵CSV')
    args = parser.parse_args()
    topic_df = pd.read_csv(args.topic_file)
    sent_df = pd.read_csv(args.sentiment_file)
    # 假定两文件有相同索引或id列
    df = topic_df.join(sent_df, lsuffix='_topic', rsuffix='_sent')
    matrix = pd.crosstab(df['topic'], df['sentiment_label'])
    matrix.to_csv(args.output)
    print(f'主题-情感矩阵已保存到 {args.output}')
    # 可选：输出每个主题的代表样本
    reps = {}
    for t in matrix.index:
        samples = df[df['topic']==t].sample(n=min(5, (df['topic']==t).sum()), random_state=42)
        reps[t] = samples['text'].tolist() if 'text' in samples.columns else samples.iloc[:,0].tolist()
    with open(args.output.replace('.csv','_samples.json'), 'w', encoding='utf-8') as f:
        json.dump(reps, f, ensure_ascii=False, indent=2)
    print(f'主题代表样本已保存到 {args.output.replace('.csv','_samples.json')}')

if __name__ == '__main__':
    main()
