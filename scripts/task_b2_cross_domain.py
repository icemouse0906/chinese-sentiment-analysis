#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务B2：跨域泛化评测
- 构建3×3跨域迁移矩阵（酒店→电商、酒店→外卖、外卖→酒店等9种组合）
- 目标域宏F1 ≥ 源域80%
- 列出≥5条典型误判样本
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import jieba
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
np.random.seed(SEED)


def load_dataset(dataset_name):
    """加载数据集"""
    if dataset_name == 'chnsenticorp':
        csv_path = ROOT / 'NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv'
    elif dataset_name == 'waimai10k':
        csv_path = ROOT / 'NLP数据集/外卖评论数据/waimai_10k.csv'
    elif dataset_name == 'ecommerce':
        csv_path = ROOT / 'NLP数据集/电商评论数据/online_shopping_10_cats.csv'
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


def split_data(df, test_size=0.2, random_state=42):
    """8/2划分（源域全量训练，目标域测试）"""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_and_evaluate_cross_domain(source_name, target_name, model_type='svm'):
    """跨域训练与评测"""
    print(f"\n{'='*60}")
    print(f"源域: {source_name} → 目标域: {target_name} | 模型: {model_type.upper()}")
    print(f"{'='*60}")
    
    # 加载数据
    source_df = load_dataset(source_name)
    target_df = load_dataset(target_name)
    
    print(f"源域样本数: {len(source_df)}")
    print(f"目标域样本数: {len(target_df)}")
    
    # 分词
    source_df['text_seg'] = source_df['text'].apply(tokenize_chinese)
    target_df['text_seg'] = target_df['text'].apply(tokenize_chinese)
    
    # 源域全量训练，目标域全量测试
    X_source = source_df['text_seg'].values
    y_source = source_df['label'].values
    X_target = target_df['text_seg'].values
    y_target = target_df['label'].values
    
    # 特征提取（在源域上拟合）
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_source_vec = vectorizer.fit_transform(X_source)
    X_target_vec = vectorizer.transform(X_target)
    
    # 训练模型
    if model_type == 'svm':
        model = LinearSVC(random_state=SEED, max_iter=2000)
    else:  # nb
        model = MultinomialNB()
    
    model.fit(X_source_vec, y_source)
    
    # 源域性能（训练集）
    y_source_pred = model.predict(X_source_vec)
    source_acc = accuracy_score(y_source, y_source_pred)
    source_f1 = f1_score(y_source, y_source_pred, average='macro')
    
    # 目标域性能（测试集）
    y_target_pred = model.predict(X_target_vec)
    target_acc = accuracy_score(y_target, y_target_pred)
    target_f1 = f1_score(y_target, y_target_pred, average='macro')
    
    print(f"\n源域性能 (训练集):")
    print(f"  准确率: {source_acc:.4f}")
    print(f"  宏F1: {source_f1:.4f}")
    
    print(f"\n目标域性能 (测试集):")
    print(f"  准确率: {target_acc:.4f}")
    print(f"  宏F1: {target_f1:.4f}")
    print(f"  F1保留率: {target_f1 / source_f1 * 100:.2f}%")
    
    # 找出典型误判样本
    errors_idx = np.where(y_target != y_target_pred)[0]
    
    error_samples = []
    for idx in errors_idx[:10]:  # 取前10个误判样本
        error_samples.append({
            'text': target_df.iloc[idx]['text'],
            'true_label': y_target[idx],
            'pred_label': y_target_pred[idx]
        })
    
    return {
        'source': source_name,
        'target': target_name,
        'model': model_type,
        'source_acc': source_acc,
        'source_f1': source_f1,
        'target_acc': target_acc,
        'target_f1': target_f1,
        'f1_retention': target_f1 / source_f1,
        'error_samples': error_samples
    }


def plot_transfer_matrix(results_df, output_path):
    """绘制3×3迁移矩阵热力图"""
    domains = ['chnsenticorp', 'waimai10k', 'ecommerce']
    domain_labels = ['酒店', '外卖', '电商']
    
    # 创建矩阵
    matrix = np.zeros((3, 3))
    
    for i, source in enumerate(domains):
        for j, target in enumerate(domains):
            row = results_df[(results_df['source'] == source) & (results_df['target'] == target)]
            if len(row) > 0:
                matrix[i, j] = row['target_f1'].values[0]
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # 设置刻度
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(domain_labels, fontsize=12)
    ax.set_yticklabels(domain_labels, fontsize=12)
    
    # 标签
    ax.set_xlabel('目标域', fontsize=13, fontweight='bold')
    ax.set_ylabel('源域', fontsize=13, fontweight='bold')
    ax.set_title('跨域迁移性能矩阵 (宏F1)', fontsize=14, fontweight='bold', pad=20)
    
    # 添加数值标注
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('宏F1 (Macro F1)', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n✓ 保存迁移矩阵: {output_path}")


def main():
    """主函数"""
    print(f"\n任务B2：跨域泛化评测")
    print(f"{'='*60}")
    
    # 禁用jieba警告
    jieba.setLogLevel(jieba.logging.INFO)
    
    # 定义域
    domains = ['chnsenticorp', 'waimai10k', 'ecommerce']
    domain_names = {'chnsenticorp': '酒店', 'waimai10k': '外卖', 'ecommerce': '电商'}
    
    # 运行所有跨域实验
    results = []
    all_error_samples = []
    
    for source in domains:
        for target in domains:
            result = train_and_evaluate_cross_domain(source, target, model_type='svm')
            results.append(result)
            
            # 收集误判样本
            for sample in result['error_samples'][:2]:  # 每个组合取2条
                all_error_samples.append({
                    'source_domain': domain_names[source],
                    'target_domain': domain_names[target],
                    'text': sample['text'],
                    'true_label': '正面' if sample['true_label'] == 1 else '负面',
                    'pred_label': '正面' if sample['pred_label'] == 1 else '负面'
                })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 创建输出目录
    output_dir = ROOT / 'results' / 'cross_domain'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存结果
    results_df.to_csv(output_dir / 'transfer_matrix.csv', index=False)
    print(f"\n✓ 保存迁移矩阵数据: {output_dir / 'transfer_matrix.csv'}")
    
    # 保存误判样本
    error_samples_df = pd.DataFrame(all_error_samples)
    error_samples_df.to_csv(output_dir / 'error_samples.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 保存误判样本: {output_dir / 'error_samples.csv'}")
    
    # 绘制迁移矩阵
    plot_transfer_matrix(results_df, output_dir / 'transfer_matrix.png')
    
    # 统计分析
    print(f"\n{'='*60}")
    print("跨域迁移性能分析")
    print(f"{'='*60}")
    
    # 同域性能
    same_domain = results_df[results_df['source'] == results_df['target']]
    print(f"\n同域性能 (源域=目标域):")
    for _, row in same_domain.iterrows():
        print(f"  {domain_names[row['source']]}: 宏F1 = {row['target_f1']:.4f}")
    
    # 跨域性能
    cross_domain = results_df[results_df['source'] != results_df['target']]
    print(f"\n跨域性能 (源域≠目标域):")
    print(f"  平均宏F1: {cross_domain['target_f1'].mean():.4f}")
    print(f"  平均F1保留率: {cross_domain['f1_retention'].mean() * 100:.2f}%")
    
    # 最佳/最差迁移
    best_transfer = cross_domain.loc[cross_domain['target_f1'].idxmax()]
    worst_transfer = cross_domain.loc[cross_domain['target_f1'].idxmin()]
    
    print(f"\n最佳迁移:")
    print(f"  {domain_names[best_transfer['source']]} → {domain_names[best_transfer['target']]}: 宏F1 = {best_transfer['target_f1']:.4f} ({best_transfer['f1_retention'] * 100:.2f}%)")
    
    print(f"\n最差迁移:")
    print(f"  {domain_names[worst_transfer['source']]} → {domain_names[worst_transfer['target']]}: 宏F1 = {worst_transfer['target_f1']:.4f} ({worst_transfer['f1_retention'] * 100:.2f}%)")
    
    # 检查是否达标
    print(f"\n达标情况 (目标域宏F1 ≥ 源域80%):")
    for _, row in cross_domain.iterrows():
        status = "✓" if row['f1_retention'] >= 0.8 else "✗"
        print(f"  {status} {domain_names[row['source']]} → {domain_names[row['target']]}: {row['f1_retention'] * 100:.2f}%")
    
    print(f"\n{'='*60}")
    print("任务B2完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
