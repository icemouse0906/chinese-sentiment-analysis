#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务A3：LDA质量度量与选K
- 对正/负样本分别做 K∈{5,10,15,20} 的 c_v coherence 扫描
- 输出每个主题的 Top-10 关键词与 3 条代表样本句
- 生成 k_sweep_coherence.png 曲线图
- README 增选K理由
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import jieba
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
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


def tokenize_text(text):
    """分词"""
    # 停用词
    stopwords = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
    
    tokens = jieba.lcut(str(text))
    tokens = [w for w in tokens if len(w) > 1 and w not in stopwords]
    return tokens


def train_lda_and_measure_coherence(texts, k, random_state=42):
    """训练LDA并计算coherence"""
    # 分词
    tokenized = [tokenize_text(t) for t in texts]
    
    # 创建词典和语料库
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    
    # 训练LDA
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=random_state,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    # 计算 coherence (c_v)
    coherence_model = CoherenceModel(
        model=lda,
        texts=tokenized,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence = coherence_model.get_coherence()
    
    return lda, dictionary, corpus, tokenized, coherence


def get_representative_docs(lda, corpus, texts, topic_id, top_n=3):
    """获取主题的代表性文档"""
    # 计算每个文档对该主题的贡献度
    doc_topic_probs = []
    for i, doc_bow in enumerate(corpus):
        topic_dist = lda.get_document_topics(doc_bow, minimum_probability=0)
        topic_prob = dict(topic_dist).get(topic_id, 0)
        doc_topic_probs.append((i, topic_prob))
    
    # 排序并取前N个
    doc_topic_probs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [texts[i] for i, _ in doc_topic_probs[:top_n]]
    
    return top_docs


def sweep_k_values(df_subset, sentiment_label, k_values=[5, 10, 15, 20]):
    """扫描不同K值的coherence"""
    print(f"\n{'='*60}")
    print(f"情感标签: {'正面' if sentiment_label == 1 else '负面'}")
    print(f"{'='*60}")
    
    texts = df_subset['text'].tolist()
    results = []
    best_k = None
    best_coherence = -float('inf')
    
    for k in k_values:
        print(f"\n训练 LDA (K={k})...")
        lda, dictionary, corpus, tokenized, coherence = train_lda_and_measure_coherence(texts, k, SEED)
        
        print(f"  Coherence (c_v): {coherence:.4f}")
        results.append({
            'k': k,
            'coherence': coherence,
            'lda': lda,
            'dictionary': dictionary,
            'corpus': corpus,
            'tokenized': tokenized
        })
        
        if coherence > best_coherence:
            best_coherence = coherence
            best_k = k
    
    print(f"\n最佳 K: {best_k} (Coherence: {best_coherence:.4f})")
    
    return results, best_k


def save_topics_info(lda, dictionary, corpus, texts, sentiment_label, k, output_dir):
    """保存主题信息"""
    topics_data = []
    
    for topic_id in range(k):
        # Top-10 关键词
        top_words = lda.show_topic(topic_id, topn=10)
        keywords = ', '.join([word for word, _ in top_words])
        
        # 代表性样本句
        rep_docs = get_representative_docs(lda, corpus, texts, topic_id, top_n=3)
        
        topics_data.append({
            'topic_id': topic_id,
            'keywords': keywords,
            'rep_doc_1': rep_docs[0] if len(rep_docs) > 0 else '',
            'rep_doc_2': rep_docs[1] if len(rep_docs) > 1 else '',
            'rep_doc_3': rep_docs[2] if len(rep_docs) > 2 else ''
        })
    
    # 保存为CSV
    topics_df = pd.DataFrame(topics_data)
    sentiment_str = 'pos' if sentiment_label == 1 else 'neg'
    output_path = output_dir / f'topics_{sentiment_str}_k{k}.csv'
    topics_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ 保存主题信息: {output_path}")


def main():
    """主函数"""
    print(f"\n任务A3：LDA质量度量与选K")
    print(f"{'='*60}")
    
    # 加载数据
    df = load_dataset('chnsenticorp')
    print(f"加载数据：{len(df)}条样本")
    
    # 分离正负样本
    pos_df = df[df['label'] == 1].reset_index(drop=True)
    neg_df = df[df['label'] == 0].reset_index(drop=True)
    
    print(f"正样本：{len(pos_df)}条")
    print(f"负样本：{len(neg_df)}条")
    
    # 创建输出目录
    output_dir = ROOT / 'results' / 'lda'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # K值范围
    k_values = [5, 10, 15, 20]
    
    # 扫描正样本
    pos_results, pos_best_k = sweep_k_values(pos_df, 1, k_values)
    
    # 扫描负样本
    neg_results, neg_best_k = sweep_k_values(neg_df, 0, k_values)
    
    # 绘制 coherence 曲线
    plt.figure(figsize=(10, 6))
    
    pos_coherences = [r['coherence'] for r in pos_results]
    neg_coherences = [r['coherence'] for r in neg_results]
    
    plt.plot(k_values, pos_coherences, marker='o', linewidth=2, label='正面样本 (Positive)', color='green')
    plt.plot(k_values, neg_coherences, marker='s', linewidth=2, label='负面样本 (Negative)', color='red')
    
    plt.xlabel('主题数量 (K)', fontsize=12)
    plt.ylabel('Coherence Score (c_v)', fontsize=12)
    plt.title('LDA主题数量与Coherence关系', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    curve_path = output_dir / 'k_sweep_coherence.png'
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"\n✓ 保存Coherence曲线: {curve_path}")
    
    # 保存最佳K的主题信息
    print(f"\n保存最佳K的主题信息...")
    
    # 正样本
    pos_best = next(r for r in pos_results if r['k'] == pos_best_k)
    save_topics_info(
        pos_best['lda'], 
        pos_best['dictionary'], 
        pos_best['corpus'], 
        pos_df['text'].tolist(), 
        1, 
        pos_best_k, 
        output_dir
    )
    
    # 负样本
    neg_best = next(r for r in neg_results if r['k'] == neg_best_k)
    save_topics_info(
        neg_best['lda'], 
        neg_best['dictionary'], 
        neg_best['corpus'], 
        neg_df['text'].tolist(), 
        0, 
        neg_best_k, 
        output_dir
    )
    
    # 保存汇总结果
    summary_data = []
    for r in pos_results:
        summary_data.append({'sentiment': 'positive', 'k': r['k'], 'coherence': r['coherence']})
    for r in neg_results:
        summary_data.append({'sentiment': 'negative', 'k': r['k'], 'coherence': r['coherence']})
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'k_sweep_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ 保存汇总结果: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"任务A3完成！")
    print(f"正面样本最佳K: {pos_best_k}")
    print(f"负面样本最佳K: {neg_best_k}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
