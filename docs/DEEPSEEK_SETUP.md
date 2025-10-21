# DeepSeek API配置说明

## 获取API密钥

1. 访问 [DeepSeek开放平台](https://platform.deepseek.com/)
2. 注册/登录账号
3. 在"API Keys"页面创建新的API密钥
4. 复制密钥（格式：`sk-...`）

## 配置方式

### 方式1：环境变量（推荐）

在终端中设置：

```bash
# macOS/Linux
export DEEPSEEK_API_KEY='sk-your-api-key-here'

# 永久保存（添加到 ~/.zshrc 或 ~/.bashrc）
echo 'export DEEPSEEK_API_KEY="sk-your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 方式2：代码中传入

```python
from scripts.task_d1_deepseek_fewshot import DeepSeekClassifier

classifier = DeepSeekClassifier(api_key='sk-your-api-key-here')
```

### 方式3：.env文件

创建 `.env` 文件（已在.gitignore中）：

```bash
DEEPSEEK_API_KEY=sk-your-api-key-here
```

然后在代码中加载：

```python
from dotenv import load_dotenv
load_dotenv()
```

## 快速开始

### 1. 安装依赖

```bash
pip install openai python-dotenv
```

### 2. Zero-shot评估

```bash
python scripts/task_d1_deepseek_fewshot.py \
  --dataset chnsenticorp \
  --shot-type zero-shot \
  --sample-size 50
```

### 3. Few-shot评估

```bash
python scripts/task_d1_deepseek_fewshot.py \
  --dataset chnsenticorp \
  --shot-type few-shot \
  --n-examples 5 \
  --sample-size 100
```

### 4. 对比两个数据集

```bash
# 酒店评论
python scripts/task_d1_deepseek_fewshot.py \
  --dataset chnsenticorp \
  --shot-type few-shot \
  --sample-size 100

# 外卖评论
python scripts/task_d1_deepseek_fewshot.py \
  --dataset waimai10k \
  --shot-type few-shot \
  --sample-size 100
```

## API计费

- DeepSeek-V3模型定价：约 ¥1/百万tokens（输入）
- 每条评论约100 tokens
- 100条样本测试成本约 ¥0.01-0.02

## 常见问题

### Q1: API调用失败？

**A:** 检查：
1. API密钥是否正确
2. 网络连接是否正常
3. 账户余额是否充足

### Q2: 输出格式不稳定？

**A:** 已设置 `temperature=0.1` 降低随机性，如仍不稳定可：
- 降低到 `temperature=0`
- 优化提示词模板
- 增加Few-shot示例

### Q3: 速度太慢？

**A:** 
- 减少 `--sample-size`
- 调整 `time.sleep()` 间隔（注意API限流）
- 使用批量API（如支持）

## 结果说明

评估完成后，结果保存在：

```
results/deepseek/{dataset}/{shot_type}/
├── classification_report.csv  # 分类报告
├── predictions.csv            # 预测结果
└── summary.json              # 性能汇总
```

## 示例输出

```
==================================================================
DeepSeek FEW-SHOT 情感分类评估
数据集: chnsenticorp | 测试样本: 100
==================================================================

📊 加载测试数据...
   测试集大小: 100 条
   正负样本: 50 / 50

🚀 初始化DeepSeek分类器...
   ✅ 连接成功

📝 准备 5 个Few-shot示例...
   示例样本:
   - [1] 酒店很不错，服务态度很好...
   - [0] 房间太小，性价比不高...

🔮 开始预测...
   预测进度: 100%|████████████| 100/100 [00:50<00:00,  1.98it/s]

   有效预测: 98 / 100

==================================================================
📊 评估结果
==================================================================
准确率 (Accuracy):    0.8673
宏F1 (Macro F1):      0.8654
加权F1 (Weighted F1): 0.8665

详细分类报告:
              precision    recall  f1-score   support
负面            0.8824    0.8491    0.8654      50
正面            0.8511    0.8854    0.8679      48

✅ 结果已保存至: results/deepseek/chnsenticorp/few-shot

🎉 评估完成!
```
