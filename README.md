
# 中文情感分析与主题建模实验

<p align="center">
	<img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python">
	<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
	<img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
	<img src="https://img.shields.io/badge/Auto--Report-Docx-blueviolet?logo=microsoftword" alt="Auto Report">
	<img src="https://img.shields.io/badge/Data-Chinese%20Reviews-orange" alt="Data">
</p>

---

## 许可与数据来源

- 本项目采用 MIT License，详见仓库根目录 LICENSE 文件。
- 所用数据集仅用于教学/研究目的，部分数据来自公开竞赛或学术资源：
	- [ChnSentiCorp](https://github.com/pengming617/bert_classification)（酒店评论）
	- [online_shopping_10_cats](https://github.com/brightmart/nlp_chinese_corpus)（电商评论）
	- [waimai_10k](https://github.com/SophonPlus/ChineseNlpCorpus)（外卖评论）
- 请勿将数据用于商业用途，使用时请遵守原数据集许可。

---


# 中文情感分析与主题建模实验


---

## 项目简介 | Project Introduction

本项目针对中文电商、酒店、外卖评论数据，完成了数据检查、预处理、情感分析、分词、LDA主题建模等实验任务，并自动生成实验报告。适用于NLP课程实验、文本分类与主题建模入门。

This project provides a full pipeline for Chinese sentiment analysis and topic modeling on e-commerce, hotel, and takeaway review datasets. It includes data inspection, preprocessing, sentiment labeling, classification, LDA topic modeling, and automated report generation.

## 目录结构 | Directory Structure


```
├── NLP数据集/                # 原始数据集
│   ├── 电商评论数据/
│   │   └── online_shopping_10_cats.csv
│   ├── 酒店评论数据/
│   │   └── ChnSentiCorp_htl_all.csv
│   └── 外卖评论数据/
│       └── waimai_10k.csv
├── scripts/                  # 主要脚本
│   ├── 01_inspect_datasets.py
│   ├── 02_preprocess_and_eda.py
│   ├── 03_label_and_model.py
│   └── 04_lda_by_sentiment.py
├── output/                   # 中间与最终输出
│   ├── samples_inspection.csv
│   ├── eda_stats.csv
│   ├── length_hist_*.png
│   ├── labels_*.csv
│   ├── classification_report_*.txt
│   └── lda_topics_*.txt
├── results/                  # 结果汇总（如有）
│   ├── classification_report.csv   # 分层评测指标
│   ├── confusion_matrix.csv        # 混淆矩阵
│   ├── pr_curve.png                # PR曲线
├── experiment_report_final.docx  # 自动生成的实验报告
├── requirements.txt
└── README.md
```

---

## 评测结果汇总 | Results Summary

### Results (True Labels) - 任务A1：真标签评测基线

**汇总指标表（ChnSentiCorp & Waimai10k）：**

| 数据集        | 模型  | 准确率 (Acc) | 宏F1 (Macro F1) | 加权F1 (Weighted F1) |
|---------------|-------|--------------|-----------------|---------------------|
| ChnSentiCorp  | NB    | 0.6324       | 0.5975          | 0.5975              |
| ChnSentiCorp  | SVM   | 0.6400       | 0.6265          | 0.6265              |
| Waimai10k     | NB    | 0.5901       | 0.5553          | 0.5553              |
| Waimai10k     | SVM   | 0.6574       | 0.6300          | 0.6300              |

**详细结果文件路径：**
- ChnSentiCorp NB: `results/chnsenticorp/nb/` (classification_report.csv, confusion_matrix.png, pr_curve.png)
- ChnSentiCorp SVM: `results/chnsenticorp/svm/` (classification_report.csv, confusion_matrix.png, pr_curve.png)
- Waimai10k NB: `results/waimai10k/nb/` (classification_report.csv, confusion_matrix.png, pr_curve.png)
- Waimai10k SVM: `results/waimai10k/svm/` (classification_report.csv, confusion_matrix.png, pr_curve.png)

**混淆矩阵与PR曲线示例：**

ChnSentiCorp - SVM Confusion Matrix:
![](results/chnsenticorp/svm/confusion_matrix.png)

ChnSentiCorp - SVM PR Curve:
![](results/chnsenticorp/svm/pr_curve.png)

Waimai10k - SVM Confusion Matrix:
![](results/waimai10k/svm/confusion_matrix.png)

Waimai10k - SVM PR Curve:
![](results/waimai10k/svm/pr_curve.png)

---

### Results (Weak Labels) - 伪标签实验（历史结果）

（此处展示之前使用SnowNLP等弱监督方法的结果，与上述真标签基线形成对照）

详见 `results/` 目录中的历史评测结果文件。

---


## 数据集说明 | Dataset Description

本项目**不包含任何原始数据集文件**，所有数据需按下述外链自行下载：

- 电商评论数据：[online_shopping_10_cats.csv](https://github.com/brightmart/nlp_chinese_corpus)  
	- 体量：约7.3MB，10分类，含商品评论文本与标签  
	- 校验和（示例，详见原仓库）：`md5: 2e5e7b1e...`  

- 酒店评论数据：[ChnSentiCorp_htl_all.csv](https://github.com/pengming617/bert_classification)  
	- 体量：约1.9MB，二分类，含酒店评论文本与标签  
	- 校验和（示例，详见原仓库）：`md5: 1a2b3c4d...`  

- 外卖评论数据：[waimai_10k.csv](https://github.com/SophonPlus/ChineseNlpCorpus)  
	- 体量：约603KB，二分类，含外卖评论文本与标签  
	- 校验和（示例，详见原仓库）：`md5: 9f8e7d6c...`  

下载后请**手动放置于 `NLP数据集/` 对应子目录**，如需校验完整性请参考原数据仓库的md5/sha1校验和。

每个数据集均为CSV格式，包含文本内容及标签（部分需自动生成）。

## 依赖环境 | Requirements

- Python 3.8+
- 推荐使用虚拟环境（venv/conda）
- 依赖安装：

```bash
pip install -r requirements.txt
```



## 环境创建与复现流程 | Environment & Reproducibility

推荐使用conda或venv创建隔离环境：

```bash
# 使用conda
conda create -n sentiment python=3.8
conda activate sentiment
pip install -r requirements.txt

# 或使用venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

完整复现流程：
1. 下载数据集并放置到 NLP数据集/ 目录
2. 安装依赖
3. 一键运行主流程（见下方命令）
4. 查看 output/ 目录结果与自动生成的实验报告

## 项目流程图 | Project Flowchart

```mermaid
graph TD
	A[数据集下载] --> B[数据检查/预处理]
	B --> C[伪标签生成/真标签分流]
	C --> D[分类模型训练与评测]
	D --> E[LDA主题建模]
	E --> F[自动化实验报告]
```

1. 创建并激活Python环境
2. 安装依赖
3. 下载数据集并放置到 `NLP数据集/` 目录
4. 一键运行主流程（推荐）：

```bash
python run.py --dataset hotel --model nb --seed 42 --mode pseudo --stage report
```

参数说明：
- --dataset [hotel|ecommerce|waimai] 选择数据集
- --model [nb|svm] 选择分类模型
- --seed 随机种子，保证可复现
- --mode [true|pseudo] 标签模式（真标签/伪标签）
- --stage [eda|label|train|lda|report] 流程阶段

如需分步运行，可分别指定 --stage 参数。

输出：控制台显示进度，结果写入 output/ 目录。



## LDA主题建模质量与可视化建议

- 主题数（K）建议通过 grid search + coherence 指标（如 c_v）选取，避免主观指定。
- 推荐使用 pyLDAVis 进行交互式主题可视化，便于理解每个主题的关键词分布和代表样本。
- 可在 output/ 目录保存 pyLDAVis HTML 文件，或在报告中插入主题-样本对照表。


1. 数据集结构检查与样本展示
	```bash
	python scripts/01_inspect_datasets.py
	```
	- 自动检测编码，展示每个数据集的行数、列名、样本，并导出样本预览表。

2. 分词、统计与可视化（预处理与EDA）
	```bash
	python scripts/02_preprocess_and_eda.py
	```
	- 分词、去停用词，统计文本长度，生成分布直方图，输出处理后CSV。

3. 情感伪标签生成与基线分类
	```bash
	python scripts/03_label_and_model.py
	```
	- 用SnowNLP自动打分，生成正负标签，训练朴素贝叶斯和SVM，输出分类报告。

4. LDA主题建模（正/负样本分开）
	```bash
	python scripts/04_lda_by_sentiment.py
	```
	- 对正负样本分别做LDA主题分析，输出每类主题关键词。

5. 自动化全流程与实验报告生成
	```bash
	python scripts/10_balance_and_eda_report.py
	```
	- 自动完成数据平衡、EDA、模型调参、评估与实验报告（docx）生成。



## 评测规范 | Evaluation Protocol

- **真标签评测**：对于自带标签的数据集（如 ChnSentiCorp、waimai_10k），请严格 train/valid/test 划分，禁止在全量数据上做重采样和EDA后再切分，避免数据泄漏。
- **伪标签实验**：仅用于无标签数据集或弱监督探索，结果仅供参考。
- **指标输出**：所有分类模型均输出 precision、recall、F1、混淆矩阵、分类报告（classification_report），并保存到 output/ 目录。
- **报告分栏**：实验报告中将真标签评测与伪标签实验分栏呈现，便于对比。

## 结果与报告 | Results & Report

- 所有中间结果、可视化图、分类报告、主题分析等均输出到`output/`目录。
- 自动生成的实验报告`experiment_report_final.docx`，包含三类数据集的完整分析与模型评估。

### 代表性结果展示

类别分布示例：

<p align="center">
	<img src="results/电商_label_distribution.png" alt="电商类别分布" width="350">
</p>

模型混淆矩阵示例：

<p align="center">
	<img src="results/电商_NB_cm.png" alt="电商NB混淆矩阵" width="350">
</p>

## 参考与致谢 | Reference & Acknowledgement

- 部分数据集来源于公开NLP竞赛与学术资源
- 代码参考了scikit-learn、pandas、snownlp等开源项目

---
如有问题欢迎提issue或PR！