# MODEL CARD

## 模型名称
中文电商/外卖/酒店评论情感分析（SVM/Transformer/弱监督融合）

## 版本
v0.1

## 结构与方法
- SVM/Naive Bayes基线
- RoBERTa-wwm/MacBERT轻量Transformer
- Snorkel式弱监督融合（9个Labeling Functions，准确率加权投票）
- 校准与拒识（Platt Scaling）
- 句向量检索（bge-small-zh-v1.5）
- 数据增强（EDA/同义词/过采样，未达标）

## 训练数据
- ChnSentiCorp酒店评论（标注）
- Waimai10k外卖评论（无标注，自动标签/弱监督）
- 电商评论（可选，未主测）

## 评测指标
- SVM基线宏F1: 0.6300
- Transformer宏F1: 0.7068（+12%）
- 弱监督宏F1: 0.7171（比伪标提升5.81pts）
- 校准拒识提升: +7.02%准确率
- 跨域F1保留率: 87.42%
- 数据增强未达标（详见C3）

## 局限性
- 数据增强对小样本提升有限，噪声易引入
- 句向量检索Top-5满意率依赖人工评估
- 仅支持二分类（正/负），中性/细粒度未覆盖
- 依赖jieba分词、snownlp、transformers等第三方库

## 适用场景
- 中文评论情感分析（电商/外卖/酒店）
- 需要可解释性/弱监督/跨域泛化的场景

## 推荐使用方式
- 通过run.py统一入口运行各任务
- 环境依赖已锁定（requirements-locked.txt）
- 复现/迁移建议用MacBERT或RoBERTa模型

## 参考/致谢
- BAAI bge-small-zh-v1.5句向量模型
- 清华NLP组ChnSentiCorp
- Snorkel弱监督思想
