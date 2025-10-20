import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os


def split_data(df, label_col, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(df, df[label_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx2, valid_idx = next(sss2.split(train_df, train_df[label_col]))
    train_final = train_df.iloc[train_idx2].reset_index(drop=True)
    valid_final = train_df.iloc[valid_idx].reset_index(drop=True)
    return train_final, valid_final, test_df


def main():
    parser = argparse.ArgumentParser(description='真标签评测与分离')
    parser.add_argument('--input', type=str, required=True, help='输入CSV，需含text和label列')
    parser.add_argument('--label_col', type=str, default='label')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # 自动适配编码
    try:
        df = pd.read_csv(args.input, encoding='utf-8')
    except Exception:
        df = pd.read_csv(args.input, encoding='gb18030')
    # 自动适配文本字段名
    text_col = None
    for col in ['text', 'review', 'content', '评论', '内容']:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        raise ValueError('未找到文本字段，请检查数据集列名。')
    train_df, valid_df, test_df = split_data(df, args.label_col, args.seed)
    train_df.to_csv(f'{args.output_dir}/train.csv', index=False)
    valid_df.to_csv(f'{args.output_dir}/valid.csv', index=False)
    test_df.to_csv(f'{args.output_dir}/test.csv', index=False)
    # 训练朴素贝叶斯
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    vec = CountVectorizer()
    X_train = vec.fit_transform(train_df[text_col])
    y_train = train_df[args.label_col]
    X_test = vec.transform(test_df[text_col])
    y_test = test_df[args.label_col]
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1] if len(clf.classes_)==2 else clf.predict_proba(X_test).max(axis=1)
    # 导出评测指标
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).to_csv(f'{args.output_dir}/classification_report.csv')
    cm = confusion_matrix(y_test, y_pred)
    np.savetxt(f'{args.output_dir}/confusion_matrix.csv', cm, delimiter=',')
    # PR曲线
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend()
    plt.savefig(f'{args.output_dir}/pr_curve.png')
    print('评测结果已导出到 results/')

if __name__ == '__main__':
    main()