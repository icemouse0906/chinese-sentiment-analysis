#!/usr/bin/env python3
"""
统一入口脚本：run.py
用法：python run.py [任务名] [可选参数]
示例：python run.py a1
        python run.py c2 --mode cli --text "很好吃"
"""
import sys
import os
import subprocess
import argparse
from pathlib import Path

TASK_MAP = {
    'a1': 'scripts/task_a1_baseline.py',
    'a2': 'scripts/task_a2_transformer.py',
    'a3': 'scripts/task_a3_lda_quality.py',
    'b1': 'scripts/task_b1_calibration.py',
    'b2': 'scripts/task_b2_cross_domain.py',
    'c1': 'scripts/task_c1_weak_supervision.py',
    'c2': 'scripts/task_c2_retrieval.py',
    'c3': 'scripts/task_c3_augmentation.py',
}


def main():
    parser = argparse.ArgumentParser(description='统一入口：运行各任务脚本')
    parser.add_argument('task', type=str, help='任务名，如a1、b2、c3')
    parser.add_argument('extra', nargs=argparse.REMAINDER, help='传递给子脚本的参数')
    args = parser.parse_args()

    task = args.task.lower()
    if task not in TASK_MAP:
        print(f"未知任务: {task}")
        print(f"可选任务: {list(TASK_MAP.keys())}")
        sys.exit(1)

    script_path = TASK_MAP[task]
    if not Path(script_path).exists():
        print(f"脚本不存在: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, script_path] + args.extra
    print(f"运行: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
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
