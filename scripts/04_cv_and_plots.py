# coding: utf-8
"""
Stratified K-Fold CV + plots + sampling for annotation
- 对每个数据集（使用 output/labels_{name}.csv）做 5 折分层 CV
- 每折训练 TF-IDF + (SVM and NB)，对每折保存：混淆矩阵、PR 曲线、ROC 曲线（用 decision_function/proba）
- 导出每折的 precision/recall/f1/accuracy，计算均值与标准差
- 为人工复核抽样保存 sample_for_annotation_{name}.csv（最多 200 条，按标签分层抽样）

运行：
python scripts/04_cv_and_plots.py

输出：
- output/cv/{name}/fold_{i}_confusion.png
- output/cv/{name}/fold_{i}_pr.png
- output/cv/{name}/fold_{i}_roc.png
- output/cv/{name}/metrics_per_fold.csv
- output/samples_for_annotation_{name}.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             roc_curve, auc, precision_score, recall_score,
                             f1_score, accuracy_score)
from sklearn.utils import resample
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
CV_OUT = OUT / 'cv'
CV_OUT.mkdir(exist_ok=True)

FILES = {
    'hotel': OUT / 'labels_hotel.csv',
    'ecommerce': OUT / 'labels_ecommerce.csv',
    'waimai': OUT / 'labels_waimai.csv'
}

N_SPLITS = 5
SAMPLE_SIZE = 200


def upsample_train(df_train, label_col='sentiment_label'):
    counts = df_train[label_col].value_counts()
    if len(counts) <= 1:
        return df_train
    maj = counts.max()
    minc = counts.min()
    if minc < max(5, maj * 0.2):
        dfs = []
        for cls, grp in df_train.groupby(label_col):
            if len(grp) < maj:
                grp_up = resample(grp, replace=True, n_samples=maj, random_state=42)
                dfs.append(grp_up)
            else:
                dfs.append(grp)
        dfb = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        return dfb
    return df_train


for name, path in FILES.items():
    if not path.exists():
        print('Missing', path)
        continue
    print('Processing CV for', name)
    df = pd.read_csv(path, encoding='utf-8', engine='python')
    # Ensure required columns exist
    if 'tokens_join' not in df.columns or 'sentiment_label' not in df.columns:
        print('File missing tokens_join or sentiment_label:', path)
        continue

    X_text = df['tokens_join'].fillna('')
    y = df['sentiment_label'].astype(int)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    cv_dir = CV_OUT / name
    cv_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []

    fold_idx = 0
    for train_idx, test_idx in skf.split(X_text, y):
        fold_idx += 1
        Xtr_text = X_text.iloc[train_idx]
        Xte_text = X_text.iloc[test_idx]
        ytr = y.iloc[train_idx]
        yte = y.iloc[test_idx]

        # upsample training if needed
        df_tr = pd.DataFrame({'text': Xtr_text.values, 'sentiment_label': ytr.values})
        df_tr_bal = upsample_train(df_tr, label_col='sentiment_label')
        Xtr_text_bal = df_tr_bal['text']
        ytr_bal = df_tr_bal['sentiment_label']

        vect = TfidfVectorizer(max_features=20000)
        Xtr = vect.fit_transform(Xtr_text_bal)
        Xte = vect.transform(Xte_text)

        # Use NB and SVM; here we'll focus on SVM for plotting but record both
        # NB
        nb = MultinomialNB()
        try:
            nb.fit(Xtr, ytr_bal)
            ypred_nb = nb.predict(Xte)
            if hasattr(nb, 'predict_proba'):
                yscore_nb = nb.predict_proba(Xte)[:, 1]
            else:
                yscore_nb = nb.predict(Xte)
        except Exception as e:
            print('NB failed on', name, 'fold', fold_idx, e)
            ypred_nb = np.zeros(len(yte), dtype=int)
            yscore_nb = np.zeros(len(yte))

        # SVM
        svm = LinearSVC(class_weight='balanced', max_iter=10000)
        try:
            svm.fit(Xtr, ytr_bal)
            ypred_svm = svm.predict(Xte)
            # decision_function for scores
            try:
                yscore_svm = svm.decision_function(Xte)
            except Exception:
                yscore_svm = ypred_svm
        except Exception as e:
            print('SVM failed on', name, 'fold', fold_idx, e)
            ypred_svm = np.zeros(len(yte), dtype=int)
            yscore_svm = np.zeros(len(yte))

        # metrics for both
        for model_name, ypred, yscore in [('nb', ypred_nb, yscore_nb), ('svm', ypred_svm, yscore_svm)]:
            prec = precision_score(yte, ypred, zero_division=0)
            rec = recall_score(yte, ypred, zero_division=0)
            f1 = f1_score(yte, ypred, zero_division=0)
            acc = accuracy_score(yte, ypred)
            # per-class precision/recall/f1 (macro) could be added; for brevity store binary scores
            metrics_rows.append({'dataset': name, 'fold': fold_idx, 'model': model_name,
                                 'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
                                 'support_pos': int((yte==1).sum()), 'support_neg': int((yte==0).sum())})

            # plots: PR and ROC
            # PR curve
            try:
                precision_vals, recall_vals, _ = precision_recall_curve(yte, yscore)
                pr_auc = auc(recall_vals, precision_vals)
                plt.figure()
                plt.plot(recall_vals, precision_vals, label=f'PR AUC={pr_auc:.4f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'{name} fold{fold_idx} {model_name} PR curve')
                plt.legend()
                plt.tight_layout()
                pr_path = cv_dir / f'fold_{fold_idx}_{model_name}_pr.png'
                plt.savefig(pr_path)
                plt.close()
            except Exception as e:
                print('Failed PR plot', name, fold_idx, model_name, e)

            # ROC curve (use yscore; if yscore is discrete may produce AUC=0.5)
            try:
                fpr, tpr, _ = roc_curve(yte, yscore)
                roc_auc = auc(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.4f}')
                plt.plot([0,1],[0,1],'k--', alpha=0.3)
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.title(f'{name} fold{fold_idx} {model_name} ROC curve')
                plt.legend()
                plt.tight_layout()
                roc_path = cv_dir / f'fold_{fold_idx}_{model_name}_roc.png'
                plt.savefig(roc_path)
                plt.close()
            except Exception as e:
                print('Failed ROC plot', name, fold_idx, model_name, e)

        # confusion matrix for SVM (main)
        try:
            cm = confusion_matrix(yte, ypred_svm)
            plt.figure(figsize=(4,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('pred')
            plt.ylabel('true')
            plt.title(f'{name} fold{fold_idx} svm confusion')
            plt.tight_layout()
            cm_path = cv_dir / f'fold_{fold_idx}_svm_confusion.png'
            plt.savefig(cm_path)
            plt.close()
        except Exception as e:
            print('Failed confusion plot', name, fold_idx, e)

    # Save metrics per fold
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(cv_dir / 'metrics_per_fold.csv', index=False, encoding='utf-8')

    # compute mean/std grouped by dataset+model
    summary = metrics_df.groupby(['dataset','model']).agg(['mean','std'])[['precision','recall','f1','accuracy']]
    # flatten columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.reset_index().to_csv(cv_dir / 'metrics_summary.csv', index=False, encoding='utf-8')

    print('Wrote CV outputs to', cv_dir)

    # Sampling for annotation: stratified sample up to SAMPLE_SIZE
    samp_dir = OUT
    sample_path = samp_dir / f'samples_for_annotation_{name}.csv'
    # stratified sampling: sample proportional but ensure representation
    try:
        n = min(SAMPLE_SIZE, len(df))
        # compute per-class counts proportional to their frequency
        value_counts = y.value_counts(normalize=True)
        draws = (value_counts * n).round().astype(int)
        # adjust to sum n
        diff = n - draws.sum()
        if diff > 0:
            # add leftover to largest class
            draws.iloc[0] += diff
        samples = []
        for cls, cnt in draws.items():
            cls_df = df[df['sentiment_label'] == cls]
            if len(cls_df) <= cnt:
                samples.append(cls_df)
            else:
                samples.append(cls_df.sample(n=cnt, random_state=42))
        sample_df = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)
        sample_df[['review','tokens_join','sentiment_score','sentiment_label']].to_csv(sample_path, index=False, encoding='utf-8')
        print('Wrote sample for annotation to', sample_path)
    except Exception as e:
        print('Failed sampling for', name, e)

print('Done all datasets')
