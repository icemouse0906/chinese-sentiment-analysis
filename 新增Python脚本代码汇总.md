# Python新增脚本代码汇总

---

## 1. FastAPI推理服务（api/predict.py）
```python
from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

app = FastAPI()

class Item(BaseModel):
    text: str

# 加载模型（示例：MacBERT）
MODEL_NAME = 'hfl/chinese-macbert-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

@app.post('/predict')
def predict(item: Item):
    inputs = tokenizer(item.text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        label = int(probs.argmax())
        conf = float(probs[label])
    return {'label': label, 'confidence': conf}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

---

## 2. 主流程统一入口（run.py）
```python
import argparse
import sys
import os
from pathlib import Path

# 支持的数据集与模式
DATASETS = {
    'hotel': 'NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv',
    'ecommerce': 'NLP数据集/电商评论数据/online_shopping_10_cats.csv',
    'waimai': 'NLP数据集/外卖评论数据/waimai_10k.csv'
}

MODELS = ['nb', 'svm']

STAGES = ['eda', 'label', 'train', 'lda', 'report']


def main():
    parser = argparse.ArgumentParser(description='统一入口：中文情感分析与主题建模实验')
    parser.add_argument('--dataset', choices=DATASETS.keys(), required=True, help='选择数据集')
    parser.add_argument('--model', choices=MODELS, default='nb', help='选择分类模型')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--mode', choices=['true', 'pseudo'], default='pseudo', help='标签模式：true=真标签，pseudo=伪标签')
    parser.add_argument('--stage', choices=STAGES, default='report', help='流程阶段')
    args = parser.parse_args()

    # 环境变量设置随机种子
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # 1. EDA与预处理
    if args.stage == 'eda':
        os.system(f'python scripts/02_preprocess_and_eda.py')
        sys.exit(0)

    # 2. 标签生成与基线分类
    if args.stage == 'label':
        os.system(f'python scripts/03_label_and_model.py')
        sys.exit(0)

    # 3. 训练与评测（伪标签/真标签分开）
    if args.stage == 'train':
        # 伪标签模式：用03脚本
        if args.mode == 'pseudo':
            os.system(f'python scripts/03_label_and_model.py')
        # 真标签模式：直接用原始数据集标签，需补充实现
        else:
            print('真标签模式暂未实现自动化，请手动运行相关脚本。')
        sys.exit(0)

    # 4. LDA主题建模
    if args.stage == 'lda':
        os.system(f'python scripts/04_lda_by_sentiment.py')
        sys.exit(0)

    # 5. 自动报告
    if args.stage == 'report':
        os.system(f'python scripts/10_balance_and_eda_report.py')
        sys.exit(0)

if __name__ == '__main__':
    main()
```

---

## 3. 数据增强脚本（scripts/augment.py）
```python
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
```

---

## 4. 跨域评测脚本（scripts/cross_domain_eval.py）
```python
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
```

---

## 5. ABSA三元组评测（scripts/eval_absa.py）
```python
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
```

---

## 6. W&B日志脚本（scripts/log_wandb.py）
```python
import wandb
import sys
import os
import json

# 用法：python scripts/log_wandb.py metrics.json
# metrics.json 格式：{"f1":0.85, "accuracy":0.88, ...}

def main():
    if len(sys.argv) < 2:
        print('用法: python scripts/log_wandb.py metrics.json')
        sys.exit(1)
    metrics_file = sys.argv[1]
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    wandb.init(project="sentiment-analysis", name=os.path.basename(metrics_file))
    wandb.log(metrics)
    wandb.finish()

if __name__ == '__main__':
    main()
```

---

## 7. 句向量生成与检索（scripts/sentence_embedding.py）
```python
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import json

def main():
    parser = argparse.ArgumentParser(description='中文句向量生成与检索')
    parser.add_argument('--input', type=str, required=True, help='输入CSV，需含text列')
    parser.add_argument('--output', type=str, default='output/sentence_embeddings.npy', help='输出npy文件')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-base-zh', help='句向量模型')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    model = SentenceTransformer(args.model_name)
    embeddings = model.encode(df['text'].tolist(), batch_size=32, show_progress_bar=True)
    np.save(args.output, embeddings)
    print(f'句向量已保存到 {args.output}')
    # Top-5相似句检索示例
    idx = 0
    query_emb = embeddings[idx]
    sims = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    top5 = np.argsort(sims)[-6:-1][::-1]
    print('Top-5相似句：')
    for i in top5:
        print(df.iloc[i]['text'])
    with open(args.output.replace('.npy','.top5.json'), 'w', encoding='utf-8') as f:
        json.dump([df.iloc[i]['text'] for i in top5], f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
```

---

## 8. 主题-情感联合分析（scripts/topic_sentiment_matrix.py）
```python
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
```

---

## 9. 强基线transformer训练（scripts/train_transformer.py）
```python
import argparse
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
import time

def main():
    parser = argparse.ArgumentParser(description='中文预训练模型基线训练')
    parser.add_argument('--model_name', type=str, default='hfl/chinese-roberta-wwm-ext', help='预训练模型名')
    parser.add_argument('--train_file', type=str, required=True, help='训练集CSV文件')
    parser.add_argument('--test_file', type=str, required=True, help='测试集CSV文件')
    parser.add_argument('--text_col', type=str, default='tokens_join', help='文本列名')
    parser.add_argument('--label_col', type=str, default='sentiment_label', help='标签列名')
    parser.add_argument('--output_dir', type=str, default='./output/transformer', help='输出目录')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # 加载数据
    import pandas as pd
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def preprocess(examples):
        return tokenizer(examples[args.text_col], truncation=True, padding='max_length', max_length=128)
    train_ds = train_ds.map(preprocess, batched=True)
    test_ds = test_ds.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        seed=args.seed,
        logging_dir=args.output_dir,
        report_to=['none'],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    start = time.time()
    trainer.train()
    end = time.time()
    print(f"训练时长: {end-start:.1f}秒")
    metrics = trainer.evaluate()
    print(metrics)
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(str(metrics))
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))

if __name__ == '__main__':
    main()
```

---

## 10. 不确定性与拒识评估（scripts/uncertainty_reject.py）
```python
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def reliability_diagram(probs, labels, output):
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
    plt.figure(figsize=(5,5))
    plt.plot(prob_pred, prob_true, marker='o', label='Reliability')
    plt.plot([0,1],[0,1],'--',label='Perfect')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.savefig(output)
    print(f'可靠性图已保存到 {output}')

def expected_calibration_error(probs, labels, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece = 0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0: continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += abs(acc-conf) * mask.sum() / len(probs)
    return ece

def main():
    parser = argparse.ArgumentParser(description='不确定性与拒识评估')
    parser.add_argument('--probs', type=str, required=True, help='预测概率CSV，需含prob,label列')
    parser.add_argument('--output', type=str, default='output/reliability.png')
    parser.add_argument('--threshold', type=float, default=0.7, help='拒识阈值')
    args = parser.parse_args()
    df = pd.read_csv(args.probs)
    probs = df['prob'].values
    labels = df['label'].values
    reliability_diagram(probs, labels, args.output)
    ece = expected_calibration_error(probs, labels)
    print(f'ECE: {ece:.4f}')
    # 拒识策略
    accept = (probs > args.threshold) | (probs < 1-args.threshold)
    acc = (df['pred']==df['label'])[accept].mean() if accept.sum()>0 else 0
    print(f'拒识后准确率: {acc:.3f}，平均审核量: {1-accept.mean():.3f}')

if __name__ == '__main__':
    main()
```

---

## 11. 弱监督标签生成（scripts/weak_labeler.py）
```python
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
```

---

（如需分模块单独复制，可直接按序号选取）
