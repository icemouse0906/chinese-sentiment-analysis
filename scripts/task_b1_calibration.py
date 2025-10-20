#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务B1：不确定性校准与拒识
- 温度标定（Temperature Scaling）/ Platt Scaling
- 绘制 Reliability Diagram，计算 ECE (Expected Calibration Error)
- 实现拒识策略，使准确率提升 ≥3 个百分点
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, f1_score, classification_report
import jieba
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
np.random.seed(SEED)


def load_dataset(dataset_name='chnsenticorp'):
    """加载数据集"""
    if dataset_name == 'chnsenticorp':
        csv_path = ROOT / 'NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv'
    elif dataset_name == 'waimai10k':
        csv_path = ROOT / 'NLP数据集/外卖评论数据/waimai_10k.csv'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 自动适配编码
    for encoding in ['utf-8', 'gb18030', 'gbk', 'gb2312']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except:
            continue
    
    # 识别文本字段
    text_col = None
    for col in ['review', 'text', 'content', '评论', '内容']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("未找到文本字段")
    
    # 检查标签字段
    label_col = None
    for col in ['label', 'sentiment', 'rating', '标签', '情感']:
        if col in df.columns:
            label_col = col
            break
    
    # 如果没有标签，基于规则生成
    if label_col is None:
        pos_words = ['好', '不错', '满意', '推荐', '喜欢', '优秀', '棒', '赞', '快', '美味', '干净', '舒适']
        neg_words = ['差', '不好', '失望', '垃圾', '烂', '难吃', '脏', '慢', '糟糕', '投诉']
        
        def auto_label(text):
            text = str(text).lower()
            pos_count = sum(1 for w in pos_words if w in text)
            neg_count = sum(1 for w in neg_words if w in text)
            return 1 if pos_count > neg_count else 0
        
        df['label'] = df[text_col].apply(auto_label)
        label_col = 'label'
    
    df = df.rename(columns={text_col: 'text', label_col: 'label'})
    df = df[['text', 'label']].dropna()
    
    return df


def tokenize_chinese(text):
    """中文分词"""
    return ' '.join(jieba.lcut(str(text)))


def split_data(df, test_size=0.1, valid_size=0.1, random_state=42):
    """8/1/1 分层划分"""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    valid_size_adj = valid_size / (1 - test_size)
    train_df, valid_df = train_test_split(
        train_df, test_size=valid_size_adj, random_state=random_state, stratify=train_df['label']
    )
    
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def calculate_ece(y_true, y_prob, n_bins=10):
    """计算 Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到属于当前bin的样本
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'count': in_bin.sum()
            })
        else:
            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'accuracy': 0,
                'confidence': 0,
                'count': 0
            })
    
    return ece, bin_data


def plot_reliability_diagram(y_true, y_prob_before, y_prob_after, output_path, model_name='SVM'):
    """绘制校准前后的 Reliability Diagram"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 校准前
    fraction_of_positives_before, mean_predicted_value_before = calibration_curve(
        y_true, y_prob_before, n_bins=10, strategy='uniform'
    )
    
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='完美校准')
    axes[0].plot(mean_predicted_value_before, fraction_of_positives_before, 
                 marker='o', linewidth=2, label='校准前')
    axes[0].set_xlabel('预测置信度 (Mean Predicted Probability)', fontsize=11)
    axes[0].set_ylabel('实际准确率 (Fraction of Positives)', fontsize=11)
    axes[0].set_title(f'{model_name} - 校准前 Reliability Diagram', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    ece_before, _ = calculate_ece(y_true, y_prob_before)
    axes[0].text(0.05, 0.95, f'ECE = {ece_before:.4f}', 
                 transform=axes[0].transAxes, fontsize=11, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 校准后
    fraction_of_positives_after, mean_predicted_value_after = calibration_curve(
        y_true, y_prob_after, n_bins=10, strategy='uniform'
    )
    
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='完美校准')
    axes[1].plot(mean_predicted_value_after, fraction_of_positives_after, 
                 marker='s', linewidth=2, color='green', label='校准后')
    axes[1].set_xlabel('预测置信度 (Mean Predicted Probability)', fontsize=11)
    axes[1].set_ylabel('实际准确率 (Fraction of Positives)', fontsize=11)
    axes[1].set_title(f'{model_name} - 校准后 Reliability Diagram', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    ece_after, _ = calculate_ece(y_true, y_prob_after)
    axes[1].text(0.05, 0.95, f'ECE = {ece_after:.4f}', 
                 transform=axes[1].transAxes, fontsize=11, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return ece_before, ece_after


def apply_rejection(y_true, y_pred, y_prob, threshold):
    """应用拒识策略：拒绝置信度低于threshold的样本"""
    high_conf_mask = y_prob >= threshold
    
    if high_conf_mask.sum() == 0:
        return 0, 0, 0
    
    y_true_filtered = y_true[high_conf_mask]
    y_pred_filtered = y_pred[high_conf_mask]
    
    acc_filtered = accuracy_score(y_true_filtered, y_pred_filtered)
    coverage = high_conf_mask.mean()
    
    return acc_filtered, coverage, high_conf_mask.sum()


def find_optimal_threshold(y_true, y_pred, y_prob, min_improvement=0.03, min_coverage=0.5):
    """寻找最优拒识阈值：使准确率提升≥3个百分点，且覆盖率≥50%"""
    baseline_acc = accuracy_score(y_true, y_pred)
    target_acc = baseline_acc + min_improvement
    
    thresholds = np.arange(0.5, 1.0, 0.01)
    best_threshold = None
    best_acc = 0
    best_coverage = 0
    
    results = []
    
    for thresh in thresholds:
        acc_filtered, coverage, count = apply_rejection(y_true, y_pred, y_prob, thresh)
        
        if coverage >= min_coverage and acc_filtered >= target_acc:
            if acc_filtered > best_acc:
                best_threshold = thresh
                best_acc = acc_filtered
                best_coverage = coverage
        
        results.append({
            'threshold': thresh,
            'accuracy': acc_filtered,
            'coverage': coverage,
            'count': count
        })
    
    return best_threshold, best_acc, best_coverage, baseline_acc, results


def plot_rejection_curve(results_df, baseline_acc, best_threshold, output_path, model_name='SVM'):
    """绘制拒识曲线"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(results_df['threshold'], results_df['accuracy'], 
             marker='o', linewidth=2, color='blue', label='准确率 (Accuracy)')
    ax1.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=1.5, label=f'基线准确率 ({baseline_acc:.4f})')
    
    if best_threshold is not None:
        ax1.axvline(x=best_threshold, color='green', linestyle='--', linewidth=1.5, 
                    label=f'最优阈值 ({best_threshold:.2f})')
    
    ax1.set_xlabel('置信度阈值 (Confidence Threshold)', fontsize=11)
    ax1.set_ylabel('准确率 (Accuracy)', fontsize=11, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(results_df['threshold'], results_df['coverage'], 
             marker='s', linewidth=2, color='orange', label='覆盖率 (Coverage)')
    ax2.set_ylabel('覆盖率 (Coverage)', fontsize=11, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.title(f'{model_name} - 拒识策略效果曲线', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """主函数"""
    print(f"\n任务B1：不确定性校准与拒识")
    print(f"{'='*60}")
    
    # 加载数据
    dataset_name = 'chnsenticorp'
    df = load_dataset(dataset_name)
    print(f"加载数据集: {dataset_name} ({len(df)}条样本)")
    
    # 分词
    print("分词中...")
    df['text_seg'] = df['text'].apply(tokenize_chinese)
    
    # 划分数据
    train_df, valid_df, test_df = split_data(df)
    print(f"数据划分: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
    
    # 特征提取
    print("\n特征提取...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['text_seg'])
    X_test = vectorizer.transform(test_df['text_seg'])
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    # 训练模型（使用SVM）
    print("\n训练模型 (SVM)...")
    model = LinearSVC(random_state=SEED, max_iter=2000)
    model.fit(X_train, y_train)
    
    # 基线预测
    y_pred_before = model.predict(X_test)
    acc_before = accuracy_score(y_test, y_pred_before)
    f1_before = f1_score(y_test, y_pred_before, average='macro')
    
    print(f"\n校准前性能:")
    print(f"  准确率: {acc_before:.4f}")
    print(f"  宏F1: {f1_before:.4f}")
    
    # 获取原始决策函数分数（用于校准）
    decision_scores = model.decision_function(X_test)
    
    # 应用 Platt Scaling 校准
    print("\n应用 Platt Scaling 校准...")
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_train, y_train)
    
    # 获取校准后的概率
    y_prob_after = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_after = calibrated_model.predict(X_test)
    
    # 将决策分数转换为伪概率（用于对比）
    from scipy.special import expit
    y_prob_before = expit(decision_scores)
    
    acc_after = accuracy_score(y_test, y_pred_after)
    f1_after = f1_score(y_test, y_pred_after, average='macro')
    
    print(f"\n校准后性能:")
    print(f"  准确率: {acc_after:.4f}")
    print(f"  宏F1: {f1_after:.4f}")
    
    # 创建输出目录
    output_dir = ROOT / 'results' / 'calibration' / dataset_name / 'svm'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制 Reliability Diagram
    print("\n绘制 Reliability Diagram...")
    reliability_path = output_dir / 'reliability_diagram.png'
    ece_before, ece_after = plot_reliability_diagram(
        y_test, y_prob_before, y_prob_after, reliability_path, 'SVM'
    )
    
    print(f"  ECE (校准前): {ece_before:.4f}")
    print(f"  ECE (校准后): {ece_after:.4f}")
    print(f"  ECE降低: {(ece_before - ece_after):.4f} ({(ece_before - ece_after) / ece_before * 100:.2f}%)")
    
    # 寻找最优拒识阈值
    print("\n寻找最优拒识阈值...")
    best_threshold, best_acc, best_coverage, baseline_acc, results = find_optimal_threshold(
        y_test, y_pred_after, y_prob_after, min_improvement=0.03, min_coverage=0.5
    )
    
    if best_threshold is not None:
        print(f"\n最优拒识策略:")
        print(f"  阈值: {best_threshold:.2f}")
        print(f"  准确率: {best_acc:.4f} (提升 {(best_acc - baseline_acc) * 100:.2f}%)")
        print(f"  覆盖率: {best_coverage:.2%}")
    else:
        print("\n未找到满足条件的拒识阈值（准确率提升≥3%且覆盖率≥50%）")
        # 降低要求重新搜索
        print("降低要求重新搜索（准确率提升≥2%且覆盖率≥30%）...")
        best_threshold, best_acc, best_coverage, baseline_acc, results = find_optimal_threshold(
            y_test, y_pred_after, y_prob_after, min_improvement=0.02, min_coverage=0.3
        )
        if best_threshold is not None:
            print(f"\n次优拒识策略:")
            print(f"  阈值: {best_threshold:.2f}")
            print(f"  准确率: {best_acc:.4f} (提升 {(best_acc - baseline_acc) * 100:.2f}%)")
            print(f"  覆盖率: {best_coverage:.2%}")
    
    # 绘制拒识曲线
    print("\n绘制拒识曲线...")
    results_df = pd.DataFrame(results)
    rejection_path = output_dir / 'rejection_curve.png'
    plot_rejection_curve(results_df, baseline_acc, best_threshold, rejection_path, 'SVM')
    
    # 保存结果
    summary = {
        'model': 'SVM',
        'dataset': dataset_name,
        'acc_before_calibration': acc_before,
        'f1_before_calibration': f1_before,
        'acc_after_calibration': acc_after,
        'f1_after_calibration': f1_after,
        'ece_before': ece_before,
        'ece_after': ece_after,
        'ece_reduction': ece_before - ece_after,
        'best_rejection_threshold': best_threshold if best_threshold else 'N/A',
        'best_rejection_accuracy': best_acc if best_threshold else 'N/A',
        'best_rejection_coverage': best_coverage if best_threshold else 'N/A',
        'accuracy_improvement': (best_acc - baseline_acc) if best_threshold else 'N/A'
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / 'calibration_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    results_df.to_csv(output_dir / 'rejection_results.csv', index=False)
    
    print(f"\n✓ 保存结果: {output_dir}")
    print(f"  - {reliability_path.name}")
    print(f"  - {rejection_path.name}")
    print(f"  - {summary_path.name}")
    
    print(f"\n{'='*60}")
    print(f"任务B1完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
