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
