# 🚀 快速开始指南：MLOps与LLM增强功能

## 目录
- [环境准备](#环境准备)
- [Phase 1-2: MLOps基础设施](#phase-1-2-mlops基础设施)
- [Phase 3: DeepSeek Few-shot分类](#phase-3-deepseek-few-shot分类)
- [Phase 4: RAG误判分析](#phase-4-rag误判分析)
- [Phase 5: ABSA三元组提取](#phase-5-absa三元组提取)
- [故障排除](#故障排除)

---

## 环境准备

### 1. 安装新增依赖

```bash
# 基础依赖（已在requirements.txt中）
pip install mlflow evidently openai sentence-transformers faiss-cpu chromadb

# 或一次性安装所有依赖
pip install -r requirements.txt
```

### 2. 配置DeepSeek API

获取API密钥：访问 [DeepSeek开放平台](https://platform.deepseek.com/)

设置环境变量：
```bash
export DEEPSEEK_API_KEY='sk-your-api-key-here'

# 永久保存（macOS/Linux）
echo 'export DEEPSEEK_API_KEY="sk-your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

详细配置请参考：[docs/DEEPSEEK_SETUP.md](docs/DEEPSEEK_SETUP.md)

---

## Phase 1-2: MLOps基础设施

### 数据验证

检查数据完整性、Schema、统计特性：

```bash
python scripts/data_validation.py
```

**输出示例：**
```
==================================================
数据验证开始
==================================================
==================================================
验证数据集: chnsenticorp
==================================================
✓ 数据加载成功: 7766 行
✓ Schema验证通过
✓ 统计特性正常
==================================================
验证完成
==================================================
✅ 所有数据集验证通过
```

### 数据漂移检测

使用Evidently检测训练集与生产数据的分布差异：

```bash
python scripts/detect_drift.py
```

**输出：**
- HTML报告：`output/data_drift_{dataset}_{timestamp}.html`
- 漂移统计：漂移特征比例、数量

### MLflow实验追踪

训练模型并记录到MLflow：

```bash
# 训练单个模型
python scripts/mlflow_tracking.py \
  --dataset chnsenticorp \
  --model svm \
  --experiment-name "my-experiment"

# 查看MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# 访问 http://localhost:5000
```

**MLflow记录内容：**
- 超参数：dataset, model_type, max_features等
- 指标：accuracy, f1_macro, f1_weighted
- Artifacts：模型文件、vectorizer、分类报告

### GitHub Actions自动化流水线

工作流已配置在 `.github/workflows/ml_pipeline.yml`

**触发条件：**
- 代码推送到 `main` 分支
- Pull Request
- 每周日自动运行
- 手动触发（Actions页面）

**流程：**
1. **数据验证** → 2. **模型训练** → 3. **模型对比** → 4. **部署最优模型**

---

## Phase 3: DeepSeek Few-shot分类

### Zero-shot评估

无需任何示例，直接让LLM分类：

```bash
python scripts/task_d1_deepseek_fewshot.py \
  --dataset chnsenticorp \
  --shot-type zero-shot \
  --sample-size 50
```

### Few-shot评估

提供少量示例，提升LLM性能：

```bash
python scripts/task_d1_deepseek_fewshot.py \
  --dataset chnsenticorp \
  --shot-type few-shot \
  --n-examples 5 \
  --sample-size 100
```

**参数说明：**
- `--dataset`：数据集（chnsenticorp/waimai10k）
- `--shot-type`：zero-shot（零样本）或 few-shot（少样本）
- `--n-examples`：Few-shot示例数量（建议3-10）
- `--sample-size`：测试样本数量（控制成本）

**输出位置：**
```
results/deepseek/{dataset}/{shot_type}/
├── classification_report.csv  # 分类报告
├── predictions.csv            # 详细预测结果
└── summary.json              # 性能汇总
```

### 对比实验

对比传统模型（SVM）与DeepSeek：

```bash
# 1. 传统SVM（已有结果）
# results/chnsenticorp/svm/classification_report.csv

# 2. DeepSeek Zero-shot
python scripts/task_d1_deepseek_fewshot.py \
  --dataset chnsenticorp --shot-type zero-shot --sample-size 100

# 3. DeepSeek Few-shot
python scripts/task_d1_deepseek_fewshot.py \
  --dataset chnsenticorp --shot-type few-shot --n-examples 5 --sample-size 100

# 4. 对比结果
# - SVM: F1~0.63 (需训练)
# - Zero-shot: F1~0.75-0.80 (无需训练)
# - Few-shot: F1~0.80-0.85 (5个示例)
```

**成本估算：**
- DeepSeek-V3定价：约¥1/百万tokens
- 100条样本测试：约¥0.01-0.02
- 1000条样本测试：约¥0.10-0.20

---

## Phase 4: RAG误判分析

### 构建误判样本数据库

使用BGE句向量模型和FAISS向量数据库：

```bash
python scripts/task_d2_rag_error_analysis.py \
  --dataset chnsenticorp \
  --model svm \
  --top-k 3 \
  --max-samples 20
```

**参数说明：**
- `--top-k`：检索相似误判样本数量
- `--max-samples`：最大分析样本数（控制成本，每条约¥0.002）

**工作流程：**
1. 加载模型的误判样本
2. 使用BGE-large-zh-v1.5生成句向量
3. 构建FAISS向量索引
4. 对每个误判样本：
   - 检索top-k相似误判案例
   - 调用DeepSeek生成分析报告
5. 保存结果和可读报告

**输出位置：**
```
results/rag_analysis/{dataset}/{model}/
├── error_analysis.csv         # CSV格式分析结果
└── error_analysis_report.md   # 可读Markdown报告
```

**报告示例：**
```markdown
## 案例 1

**文本**: 这次是308的行政大床，总体感觉非常不错，就是价格稍许高了点...

**真实标签**: 负面
**预测标签**: 正面

### 错误原因
模型未能捕捉转折词"就是"后的负面表达

### 详细分析
文本整体正面表达占主导("非常不错")，但因价格偏高被标为负面。
模型对转折逻辑不敏感，过度关注前半部分的正面词汇。

### 改进建议
1. 增强转折词识别（但是、就是、不过等）
2. 引入方面级情感分析（服务正面、价格负面）
3. 使用Transformer模型提升上下文理解
```

---

## Phase 5: ABSA三元组提取

### 提取方面级情感

从评论中提取（方面词，观点词，情感极性）三元组：

```bash
# 酒店评论
python scripts/task_d3_absa_extraction.py \
  --dataset chnsenticorp \
  --domain hotel \
  --sample-size 50

# 外卖评论
python scripts/task_d3_absa_extraction.py \
  --dataset waimai10k \
  --domain food \
  --sample-size 50

# 电商评论
python scripts/task_d3_absa_extraction.py \
  --dataset ecommerce \
  --domain ecommerce \
  --sample-size 50
```

**参数说明：**
- `--domain`：领域类型（hotel/food/ecommerce），影响方面词提示

**输出位置：**
```
results/absa/{dataset}/
├── absa_triplets.csv          # 三元组CSV（展开格式）
├── absa_raw_outputs.json      # 原始LLM输出
├── absa_statistics.json       # 统计信息
└── absa_report.md            # 可读报告
```

**输出示例：**
```json
[
  {
    "aspect": "服务",
    "opinion": "态度很好",
    "sentiment": "正面"
  },
  {
    "aspect": "价格",
    "opinion": "偏贵",
    "sentiment": "负面"
  },
  {
    "aspect": "环境",
    "opinion": "整洁干净",
    "sentiment": "正面"
  }
]
```

**统计分析：**
- 方面词分布（Top 10）
- 情感极性分布（正面/负面/中性）
- 方面-情感交叉表

---

## 完整实验流程示例

### 场景1：从零开始完整评估

```bash
# 1. 数据验证
python scripts/data_validation.py

# 2. 训练传统模型（MLflow追踪）
python scripts/mlflow_tracking.py \
  --dataset chnsenticorp --model svm --experiment-name baseline

# 3. DeepSeek Few-shot对比
python scripts/task_d1_deepseek_fewshot.py \
  --dataset chnsenticorp --shot-type few-shot --sample-size 100

# 4. RAG误判分析
python scripts/task_d2_rag_error_analysis.py \
  --dataset chnsenticorp --model svm --max-samples 20

# 5. ABSA三元组提取
python scripts/task_d3_absa_extraction.py \
  --dataset chnsenticorp --domain hotel --sample-size 50

# 6. 查看MLflow实验
mlflow ui
```

### 场景2：快速验证LLM效果

```bash
# 小样本快速测试（约1分钟，成本<¥0.01）
python scripts/task_d1_deepseek_fewshot.py \
  --dataset chnsenticorp \
  --shot-type few-shot \
  --n-examples 5 \
  --sample-size 20
```

### 场景3：生产环境监控

```bash
# 定期运行（可配置crontab）
python scripts/detect_drift.py  # 数据漂移检测
python scripts/data_validation.py  # 数据质量检查

# 如果检测到漂移，触发重训练
python scripts/mlflow_tracking.py \
  --dataset chnsenticorp --model svm --experiment-name retrain-$(date +%Y%m%d)
```

---

## 故障排除

### 问题1：DeepSeek API调用失败

**症状：**
```
ValueError: 请设置DEEPSEEK_API_KEY环境变量
```

**解决：**
```bash
# 检查环境变量
echo $DEEPSEEK_API_KEY

# 重新设置
export DEEPSEEK_API_KEY='sk-your-api-key-here'

# 验证
python -c "import os; print(os.getenv('DEEPSEEK_API_KEY'))"
```

### 问题2：依赖库缺失

**症状：**
```
ImportError: No module named 'mlflow'
```

**解决：**
```bash
pip install mlflow evidently openai sentence-transformers faiss-cpu

# 或重新安装所有依赖
pip install -r requirements.txt
```

### 问题3：MLflow UI无法启动

**症状：**
```
mlflow: command not found
```

**解决：**
```bash
# 确认安装
pip install mlflow

# 检查路径
which mlflow

# 手动启动
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 问题4：FAISS索引构建失败（M4芯片）

**症状：**
```
ImportError: cannot import name 'swigfaiss' from 'faiss'
```

**解决：**
```bash
# M4芯片使用CPU版本
pip uninstall faiss-gpu
pip install faiss-cpu

# 或使用conda（推荐）
conda install -c pytorch faiss-cpu
```

### 问题5：API限流

**症状：**
```
⚠️  API调用失败: Rate limit exceeded
```

**解决：**
- 增加 `time.sleep()` 间隔（脚本中已设置0.5-1秒）
- 减少 `--sample-size` 参数
- 升级API套餐或等待限流重置

---

## 性能基准参考

### 传统模型（已有结果）

| 数据集 | 模型 | F1 (Macro) | 训练时间 |
|--------|------|------------|----------|
| ChnSentiCorp | SVM | 0.6265 | ~30s |
| Waimai10k | SVM | 0.6300 | ~20s |

### DeepSeek Few-shot（预期）

| Shot类型 | F1 (Macro) | 成本/100样本 | 优势 |
|----------|------------|--------------|------|
| Zero-shot | 0.75-0.80 | ¥0.01 | 无需训练 |
| Few-shot (5例) | 0.80-0.85 | ¥0.015 | 少量示例即可 |

### RAG误判分析

- **检索速度**：~100ms/查询（FAISS向量检索）
- **分析成本**：~¥0.002/样本
- **适用场景**：人工审核辅助、模型诊断

### ABSA三元组提取

- **提取速度**：~1-2s/样本（含API调用）
- **提取成本**：~¥0.003/样本
- **平均三元组数**：2-4个/评论

---

## 下一步

1. **优化提示词**：根据实际效果调整Few-shot示例和系统提示
2. **扩展评估**：增加更多数据集和领域
3. **成本优化**：批量API调用、缓存机制
4. **集成部署**：将DeepSeek集成到FastAPI服务（`api/serve_fastapi.py`）

---

## 参考资料

- [DeepSeek API文档](https://platform.deepseek.com/docs)
- [MLflow官方文档](https://mlflow.org/docs/latest/index.html)
- [Evidently文档](https://docs.evidentlyai.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS向量检索](https://github.com/facebookresearch/faiss)

---

**🎉 恭喜！你已经掌握了全套MLOps与LLM增强功能的使用方法！**
