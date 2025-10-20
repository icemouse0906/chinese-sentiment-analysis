import argparse
import pandas as pd
import ast
from collections import Counter

def parse_triplets(s):
    # 支持JSON或分号分隔
    if not s or pd.isna(s):
        return []
    try:
        if s.startswith('['):
            return ast.literal_eval(s)
        return [tuple(x.split(',')) for x in s.split(';') if x]
    except Exception:
        return []

def f1_score(pred, gold):
    pred_set = set(pred)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description='ABSA三元组评测')
    parser.add_argument('--pred', type=str, required=True, help='预测结果TSV')
    parser.add_argument('--gold', type=str, required=True, help='金标TSV')
    args = parser.parse_args()
    pred_df = pd.read_csv(args.pred, sep='\t')
    gold_df = pd.read_csv(args.gold, sep='\t')
    assert len(pred_df) == len(gold_df)
    p_total, r_total, f_total = 0, 0, 0
    n = len(pred_df)
    for i in range(n):
        pred_triplets = parse_triplets(pred_df.iloc[i]['triplets'])
        gold_triplets = parse_triplets(gold_df.iloc[i]['triplets'])
        p, r, f = f1_score(pred_triplets, gold_triplets)
        p_total += p
        r_total += r
        f_total += f
    print(f'三元组F1: {f_total/n:.3f} Precision: {p_total/n:.3f} Recall: {r_total/n:.3f}')

if __name__ == '__main__':
    main()
