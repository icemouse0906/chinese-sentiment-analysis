# 实验报告（草稿）

日期：2025-10-20
作者：自动生成（请校验与补充）

## 一、摘要
本实验对三个中文评论数据集（酒店 ChnSentiCorp_htl_all、商品评论 online_shopping_10_cats、外卖评论 waimai_10k）进行了统一预处理、自动情感打分（SnowNLP）、以及基于 TF-IDF 的两种基线分类器（MultinomialNB / LinearSVC）的训练与评估。重点工作：修复编码/转义问题、统一 UTF-8 输出、在训练集上做少数类上采样及分层拆分回退策略。本文档为实验结果草稿，含关键指标、代表样例、发现与下一步建议。

## 二、数据集简要
- hotel: 原始文件 `NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv`（处理后 `data/processed_hotel.csv`）
- ecommerce: 原始 `NLP数据集/电商评论数据/online_shopping_10_cats.csv`（处理后 `data/processed_ecommerce.csv`）
- waimai: 原始 `NLP数据集/外卖评论数据/waimai_10k.csv`（处理后 `data/processed_waimai.csv`）

输出文件（重要）：
- `data/processed_{hotel,ecommerce,waimai}.csv`（UTF-8）
- `output/labels_{hotel,ecommerce,waimai}.csv`（含 sentiment_score, sentiment_label）
- `output/classification_report_{*_nb|_svm}.txt`
- `output/eda_stats.csv`, `output/length_hist_{name}.png`

## 三、预处理与打标签
- 自动编码检测并回退：先检测（chardet），尝试编码顺序为[检测值, utf-8, gb18030, latin1]，并使用 `on_bad_lines='skip'` 提高鲁棒性；最终结果用 gb18030 成功读取原始文件（日志中有记录）。
- 文本清洗（`simple_clean`）：剔除换行、尝试对包含文字型转义序列（如 `\u4e2d` / `\xE5`）进行 `unicode_escape` 解码以恢复中文。
- 分词：使用 `jieba.cut()`，生成 `tokens` 列与 `tokens_join`（以空格连接的分词结果，便于 TF-IDF）；写出 processed CSV 时强制 `encoding='utf-8'`。
- 自动标注：使用 SnowNLP 计算 `sentiment_score`（[0,1]），阈值 0.5 -> `sentiment_label`（1=正/非负，0=负）。

## 四、模型与训练流程（简述）
- 特征：TfidfVectorizer(max_features=20000) 使用 `tokens_join`。
- 模型：MultinomialNB；LinearSVC(class_weight='balanced')（若训练单类则捕获异常并记录）。
- 不平衡处理：若训练集显著不平衡（少数类小于 max(5, maj*0.2)），对训练集做上采样（resample）到多数类大小；分层拆分时在 `stratify` 不稳时尝试多次 random_state 回退，并在必要情况下显式把一个少数类样本加入测试集以保证测试覆盖。

## 五、关键结果（classification_report 摘要）
摘自 `output/classification_report_*.txt`，下列为两类分数的主要指标（precision / recall / f1 / support）：

- hotel (NB)
  - class 0: precision=0.7391 recall=0.9906 f1=0.8466 support=958
  - class 1: precision=0.9665 recall=0.4370 f1=0.6019 support=595
  - accuracy=0.7785

- hotel (SVM)
  - class 0: precision=0.9135 recall=0.8706 f1=0.8915 support=958
  - class 1: precision=0.8063 recall=0.8672 f1=0.8356 support=595
  - accuracy=0.8693

- ecommerce (NB)
  - class 0: precision=0.8666 recall=0.8999 f1=0.8829 support=6541
  - class 1: precision=0.8863 recall=0.8494 f1=0.8675 support=6014
  - accuracy=0.8757

- ecommerce (SVM)
  - class 0: precision=0.9083 recall=0.9054 f1=0.9068 support=6541
  - class 1: precision=0.8974 recall=0.9006 f1=0.8990 support=6014
  - accuracy=0.9031

- waimai (NB)
  - class 0: precision=0.8228 recall=0.9499 f1=0.8818 support=1496
  - class 1: precision=0.8882 recall=0.6608 f1=0.7578 support=902
  - accuracy=0.8411

- waimai (SVM)
  - class 0: precision=0.8969 recall=0.8663 f1=0.8813 support=1496
  - class 1: precision=0.7901 recall=0.8348 f1=0.8119 support=902
  - accuracy=0.8545

简评：
- SVM 在三套数据上均优于或匹配 NB（总体 f1/accuracy 更高），尤其在 hotel 上 SVM 大幅改善了 minority 类 recall 与 f1。 
- ecommerce 数据在两模型上已接近平衡且表现良好（类间支持数大、指标稳定）。
- waimai 的 NB 对少数类 recall 较低，SVM 提升了均衡性。

## 六、代表性样例（来源：`output/labels_*` 的 head）
（每项：文本片段 → tokens_join → sentiment_score → sentiment_label）

- hotel 示例 1:
  - 文本（节选）："距离川沙公路较近,但是公交指示不对... 房间较为简单."
  - tokens_join（示例）："距离 川沙 公路 较近 , 但是 公交 指示 对 , 如果 \"\" 蔡陆线 \"\" 的话 ..."
  - sentiment_score=0.8261 → sentiment_label=1

- ecommerce 示例 1:
  - 文本（节选）："做父母一定要有刘墉这样的心态...家庭教育，真的是乐在其中."
  - tokens_join（示例）："? 做 父母 一定 要 刘墉 这样 心态 ..."
  - sentiment_score=1.0 → sentiment_label=1

- waimai 示例 1:
  - 文本："很快，好吃，味道足，量大"
  - tokens_join："很快 ， 好吃 ， 味道 足 ， 量 大"
  - sentiment_score=0.8761 → sentiment_label=1

（注：以上示例为 head 中实际行，已确认为 UTF-8 可读中文，说明预处理写出编码问题已被修复。）

## 七、已改动的代码文件（最小化说明）
- `scripts/02_preprocess_and_eda.py`（修改）
  - 改动点：增强 CSV 读取回退（检测编码 -> utf-8 -> gb18030 -> latin1），使用 `on_bad_lines='skip'`；写出 `processed_*.csv` 与 `eda_stats.csv` 时强制 `encoding='utf-8'`；保持原有的 literal-escape 解码逻辑。
- `scripts/03_label_and_model.py`（已审阅，未改动）
  - 该脚本已采用 utf-8 写出 labels 与报告，并包含训练集上采样与分层拆分回退策略。

## 八、重现命令（在项目根、zsh）
下面命令是在虚拟环境 `.venv` 中运行的，已被执行并成功（见运行日志）。可按需手动运行以重现：

```zsh
source .venv/bin/activate
python scripts/02_preprocess_and_eda.py
python scripts/03_label_and_model.py
```

或直接（合并命令）：

```zsh
.venv/bin/python3 scripts/02_preprocess_and_eda.py && .venv/bin/python3 scripts/03_label_and_model.py
```

## 九、我运行时的关键信息（简要日志摘要）
- 读取原始文件时，chardet 常检测到 `GB2312`，但实际以 `gb18030` 成功读取（脚本会尝试并回退）：
  - "Success reading ChnSentiCorp_htl_all.csv with encoding=gb18030"
  - 同样对 ecommerce 与 waimai 成功读取为 gb18030。
- processed CSV 与 labels/报告均已写出：例如
  - `Wrote processed data to data/processed_hotel.csv`
  - `Wrote labels to output/labels_hotel.csv`
  - `Wrote classification reports for hotel` 等。

## 十、主要发现与讨论
1. 编码/转义问题已解决：此前观察到的 mojibake 与转义序列现象在最新 run 后已修复（输出文本可读）。
2. 模型表现：
   - SVM 在三套数据上均为更强的基线（f1/accuracy 更高）。
   - ecommerce 数据样本量大且两类支持均衡，模型稳定。
   - waimai 的少数类在 NB 下 recall 较低，SVM 改进明显。
3. 评估注意事项：
   - 目前各数据集的 class 支持（support）在最新拆分下都足够大（非极端 1-5 支持），因此当前 accuracy 与 f1 指标具有一定参考价值。若将来改用不同拆分/小样本测试，需注意 support 的稳定性。

## 十一、局限性
- 自动标签（SnowNLP）为弱监督/伪标签，可能带有偏差；建议将部分自动标签抽样人工复核（至少每数据集 100-300 条）以估计噪声率。
- 目前仅使用 TF-IDF + 传统分类器；后续可尝试中文预训练模型（BERT/ERNIE）或文本增强以提升泛化。

## 十二、建议的下一步（优先级）
1. 对 `labels_*.csv` 做抽样人工标注（200~500 条/数据集）用于估计 SnowNLP 伪标签噪声并作为少量真实标注用于微调模型。
2. 引入 Stratified K-Fold（例如 5 折），计算每折混淆矩阵与 PR 曲线，导出平均/标准差指标，增加评估稳健性（推荐优先执行）。
3. 比较不平衡处理方法（上采样、下采样、文本增强、代价敏感学习、阈值校准），并把重点放在 minority recall/precision 的改善上。 
4. 若需最终提交（课程/论文），我可以把本草稿整理成 PDF 并补充图表（长度分布图/混淆矩阵/PR 曲线），也可以把关键结果写入报告 Word 格式。

## 十三、结论
已完成：预处理（含编码修复）、伪标签生成、TF-IDF + NB/SVM 训练并输出分类报告。SVM 为更稳健的基线。当前结果足够用于撰写中期实验报告；如果你希望我继续，我可以继续执行 K-Fold、生成图表并形成最终版报告（含图片与混淆矩阵）。

---

如需我把本报告转换为 PDF/Word 并把它放到 `output/` 下，或继续做 5 折 CV 并生成图表，请直接回复 "生成最终报告(PDF)" 或 "做5折CV"，我会继续执行并把结果写入工作区。

## 十四、本次新增的 CV 输出与抽样（已生成）

我已对三个数据集执行 5 折分层交叉验证，并生成每折的 PR/ROC 曲线与 SVM 混淆矩阵，以及每数据集的平均/标准差指标文件。关键文件：

- `output/cv/{hotel,ecommerce,waimai}/metrics_summary.csv`（每个数据集，包含 precision/recall/f1/accuracy 的 mean/std）
- `output/cv/{name}/metrics_per_fold.csv`（每折的原始度量）
- 各折图像示例（保存在对应文件夹）：
  - `fold_{i}_svm_confusion.png`
  - `fold_{i}_svm_pr.png`
  - `fold_{i}_svm_roc.png`

此外，为每个数据集生成了供人工复核的抽样文件（最多 200 条，分层抽样）：

- `output/samples_for_annotation_hotel.csv`
- `output/samples_for_annotation_ecommerce.csv`
- `output/samples_for_annotation_waimai.csv`

下面为 CV 汇总（从 `metrics_summary.csv` 摘要）：

- hotel (nb): precision_mean=0.961 ±0.012, recall_mean=0.427 ±0.0099, f1_mean=0.591 ±0.0081, accuracy_mean=0.774 ±0.0026
- hotel (svm): precision_mean=0.807 ±0.026, recall_mean=0.844 ±0.014, f1_mean=0.825 ±0.018, accuracy_mean=0.862 ±0.016

- ecommerce (nb): precision_mean=0.886 ±0.0032, recall_mean=0.855 ±0.0043, f1_mean=0.870 ±0.0037, accuracy_mean=0.878 ±0.0034
- ecommerce (svm): precision_mean=0.899 ±0.0019, recall_mean=0.901 ±0.0058, f1_mean=0.900 ±0.0028, accuracy_mean=0.904 ±0.0024

- waimai (nb): precision_mean=0.855 ±0.0136, recall_mean=0.629 ±0.0223, f1_mean=0.725 ±0.0173, accuracy_mean=0.820 ±0.0096
- waimai (svm): precision_mean=0.788 ±0.0178, recall_mean=0.802 ±0.0169, f1_mean=0.795 ±0.0108, accuracy_mean=0.844 ±0.0088

这些结果将被包含在最终 PDF 中，并附上每折的混淆矩阵与 PR/ROC 图像作为附录。

## 十五：现在我要把此 Markdown 转成 PDF 并保存为 `output/experiment_report.pdf`（几种可能的实现路径）

我将先尝试使用系统上的 `pandoc` 将 Markdown 转为 PDF（优选 xelatex 字体支持中文）；若系统无 pandoc 或无 LaTeX，我会回退到在虚拟环境中用 Python 库生成一个包含文本和关键图片的基础 PDF（保证可阅读）。

现在我将检测系统上是否有 `pandoc` 或 `wkhtmltopdf`，以决定转换方法。后续我会运行转换并把最终 `output/experiment_report.pdf` 放到工作区。
