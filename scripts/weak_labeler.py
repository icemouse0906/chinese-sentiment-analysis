import pandas as pd
import numpy as np
import argparse
import json

# 规则示例：词典法、模型分数法、启发式投票
POS_WORDS = set(['好', '棒', '快', '新鲜', '满意', '赞', '喜欢'])
NEG_WORDS = set(['慢', '差', '坏', '脏', '失望', '破损', '不满意'])


def rule_label(text):
    # 简单词典法
    for w in POS_WORDS:
        if w in text:
            return 1
    for w in NEG_WORDS:
        if w in text:
            return 0
    return -1  # 未知

def combine_labels(row, threshold=0.8):
    # 规则优先，模型分数辅助
    rule = rule_label(row['text'])
    model_score = row.get('model_score', None)
    if rule != -1:
        return rule, 'rule'
    if model_score is not None:
        if model_score >= threshold:
            return 1, 'model_high'
        elif model_score <= 1-threshold:
            return 0, 'model_low'
    return -1, 'abstain'


def main():
    parser = argparse.ArgumentParser(description='弱监督标签生成器')
    parser.add_argument('--input', type=str, required=True, help='输入CSV，需含text列和可选model_score列')
    parser.add_argument('--output', type=str, required=True, help='输出CSV')
    parser.add_argument('--threshold', type=float, default=0.8, help='模型分数置信度阈值')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    labels = []
    sources = []
    for _, row in df.iterrows():
        label, source = combine_labels(row, args.threshold)
        labels.append(label)
        sources.append(source)
    df['weak_label'] = labels
    df['label_source'] = sources
    df.to_csv(args.output, index=False)
    print(f'已生成弱监督标签，保存到 {args.output}')

if __name__ == '__main__':
    main()
