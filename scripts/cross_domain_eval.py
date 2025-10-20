import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# 跨域评测：电商→外卖→酒店

def main():
    parser = argparse.ArgumentParser(description='跨域与公平性评测')
    parser.add_argument('--pred', type=str, required=True, help='预测结果CSV，需含pred,label列')
    parser.add_argument('--domain', type=str, required=True, help='域名')
    args = parser.parse_args()
    df = pd.read_csv(args.pred)
    macro_f1 = f1_score(df['label'], df['pred'], average='macro')
    print(f'{args.domain}域宏F1: {macro_f1:.3f}')
    # 偏置词典扫描
    bias_words = ['地区','性别','方言','表情']
    bias_mask = df['text'].apply(lambda x: any(w in str(x) for w in bias_words))
    bias_acc = (df['pred']==df['label'])[bias_mask].mean() if bias_mask.sum()>0 else 0
    print(f'偏置样本误判率: {1-bias_acc:.3f} (样本数: {bias_mask.sum()})')

if __name__ == '__main__':
    main()
