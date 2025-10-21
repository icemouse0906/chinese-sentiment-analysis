"""
Task D1: DeepSeek Few-shot情感分类
对比传统微调模型与大语言模型的Prompt Engineering效果
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
from datetime import datetime
from tqdm import tqdm
import time

class DeepSeekClassifier:
    """DeepSeek情感分类器"""
    
    def __init__(self, api_key=None, base_url="https://api.deepseek.com"):
        """
        初始化DeepSeek客户端
        
        Args:
            api_key: DeepSeek API密钥（如未提供则从环境变量读取）
            base_url: API地址
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量或传入api_key参数")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.model = "deepseek-chat"
    
    def create_few_shot_prompt(self, examples, test_text, task_type="sentiment"):
        """
        创建Few-shot提示词
        
        Args:
            examples: 示例列表 [(text, label), ...]
            test_text: 待分类文本
            task_type: 任务类型（sentiment/absa）
        """
        if task_type == "sentiment":
            system_prompt = """你是一个专业的中文情感分析助手。请根据给定的评论文本，判断其情感倾向。
输出格式：仅输出"正面"或"负面"，不要有任何额外解释。"""
            
            user_prompt = "以下是一些示例：\n\n"
            for text, label in examples:
                label_text = "正面" if label == 1 else "负面"
                user_prompt += f"评论：{text}\n情感：{label_text}\n\n"
            
            user_prompt += f"现在请判断以下评论的情感：\n评论：{test_text}\n情感："
            
        elif task_type == "absa":
            system_prompt = """你是一个专业的方面级情感分析（ABSA）助手。请从评论中提取（方面词，观点词，情感极性）三元组。
输出格式：JSON数组，例如：[{"aspect": "服务", "opinion": "态度好", "sentiment": "正面"}]"""
            
            user_prompt = f"请分析以下评论的方面级情感：\n{test_text}"
        
        return system_prompt, user_prompt
    
    def classify_sentiment(self, text, examples=None, max_retries=3):
        """
        情感分类
        
        Args:
            text: 待分类文本
            examples: Few-shot示例
            max_retries: 最大重试次数
        """
        if examples is None:
            # Zero-shot默认示例
            examples = []
        
        system_prompt, user_prompt = self.create_few_shot_prompt(examples, text, "sentiment")
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # 低温度保证稳定输出
                    max_tokens=10
                )
                
                result = response.choices[0].message.content.strip()
                
                # 解析结果
                if "正面" in result:
                    return 1, result
                elif "负面" in result:
                    return 0, result
                else:
                    # 如果输出不明确，使用更严格的匹配
                    if "positive" in result.lower() or "好" in result:
                        return 1, result
                    elif "negative" in result.lower() or "差" in result:
                        return 0, result
                    else:
                        # 重试
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        else:
                            return -1, result  # 无法判断
            
            except Exception as e:
                print(f"  ⚠️  API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return -1, str(e)
        
        return -1, "未知"
    
    def extract_absa_triplets(self, text):
        """提取ABSA三元组"""
        system_prompt = """你是一个专业的方面级情感分析助手。请从评论中提取（方面词，观点词，情感极性）三元组。
输出格式：JSON数组，例如：[{"aspect": "服务", "opinion": "态度好", "sentiment": "正面"}]
如果没有明确的方面词，请提取"整体"作为方面词。"""
        
        user_prompt = f"请分析以下评论的方面级情感：\n{text}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            # 尝试解析JSON
            try:
                triplets = json.loads(result)
                return triplets, result
            except:
                # 如果不是标准JSON，返回原始文本
                return [], result
        
        except Exception as e:
            print(f"  ⚠️  ABSA提取失败: {str(e)}")
            return [], str(e)

def load_test_data(dataset_name, sample_size=100):
    """加载测试数据"""
    base_dir = Path(__file__).parent.parent
    
    if dataset_name == 'chnsenticorp':
        data_path = base_dir / 'NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv'
    elif dataset_name == 'waimai10k':
        data_path = base_dir / 'NLP数据集/外卖评论数据/waimai_10k.csv'
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    df = pd.read_csv(data_path)
    
    # 随机采样（保持类别平衡）
    df_sampled = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 2), random_state=42)
    )
    
    return df_sampled

def evaluate_deepseek(dataset_name, shot_type='zero-shot', n_examples=5, sample_size=100):
    """
    评估DeepSeek模型性能
    
    Args:
        dataset_name: 数据集名称
        shot_type: 'zero-shot' 或 'few-shot'
        n_examples: Few-shot示例数量
        sample_size: 测试样本数量
    """
    print(f"\n{'='*70}")
    print(f"DeepSeek {shot_type.upper()} 情感分类评估".center(70))
    print(f"数据集: {dataset_name} | 测试样本: {sample_size}".center(70))
    print(f"{'='*70}\n")
    
    # 加载数据
    print("📊 加载测试数据...")
    df = load_test_data(dataset_name, sample_size)
    print(f"   测试集大小: {len(df)} 条")
    print(f"   正负样本: {(df['label']==1).sum()} / {(df['label']==0).sum()}")
    
    # 初始化分类器
    print("\n🚀 初始化DeepSeek分类器...")
    try:
        classifier = DeepSeekClassifier()
        print("   ✅ 连接成功")
    except Exception as e:
        print(f"   ❌ 初始化失败: {str(e)}")
        print("\n💡 提示：请设置环境变量 DEEPSEEK_API_KEY")
        print("   export DEEPSEEK_API_KEY='your-api-key'")
        return
    
    # 准备Few-shot示例
    examples = []
    if shot_type == 'few-shot':
        print(f"\n📝 准备 {n_examples} 个Few-shot示例...")
        # 从数据集中选择示例（不与测试集重叠）
        all_data = pd.read_csv(Path(__file__).parent.parent / f'NLP数据集/{"酒店评论数据" if dataset_name == "chnsenticorp" else "外卖评论数据"}/{"ChnSentiCorp_htl_all.csv" if dataset_name == "chnsenticorp" else "waimai_10k.csv"}')
        example_indices = df.index.tolist()
        remaining_data = all_data[~all_data.index.isin(example_indices)]
        
        # 均衡选择正负样本
        pos_examples = remaining_data[remaining_data['label'] == 1].sample(n_examples // 2, random_state=42)
        neg_examples = remaining_data[remaining_data['label'] == 0].sample(n_examples // 2, random_state=42)
        
        for _, row in pd.concat([pos_examples, neg_examples]).iterrows():
            examples.append((row['review'][:100], row['label']))  # 限制长度
        
        print(f"   示例样本:")
        for text, label in examples[:2]:
            print(f"   - [{label}] {text[:50]}...")
    
    # 批量预测
    print(f"\n🔮 开始预测...")
    predictions = []
    raw_outputs = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   预测进度"):
        pred, output = classifier.classify_sentiment(row['review'], examples)
        predictions.append(pred)
        raw_outputs.append(output)
        
        # 限速（避免API限流）
        time.sleep(0.5)
    
    # 过滤无效预测
    valid_mask = np.array(predictions) != -1
    df_valid = df[valid_mask].copy()
    predictions_valid = np.array(predictions)[valid_mask]
    
    print(f"\n   有效预测: {len(predictions_valid)} / {len(predictions)}")
    
    if len(predictions_valid) == 0:
        print("❌ 无有效预测，评估终止")
        return
    
    # 计算指标
    y_true = df_valid['label'].values
    y_pred = predictions_valid
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{'='*70}")
    print("📊 评估结果".center(70))
    print(f"{'='*70}")
    print(f"准确率 (Accuracy):    {accuracy:.4f}")
    print(f"宏F1 (Macro F1):      {f1_macro:.4f}")
    print(f"加权F1 (Weighted F1): {f1_weighted:.4f}")
    
    # 分类报告
    report = classification_report(y_true, y_pred, target_names=['负面', '正面'], output_dict=True)
    report_df = pd.DataFrame(report).T
    
    print(f"\n详细分类报告:")
    print(report_df.to_string())
    
    # 保存结果
    output_dir = Path(__file__).parent.parent / 'results' / 'deepseek' / dataset_name / shot_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存分类报告
    report_df.to_csv(output_dir / 'classification_report.csv')
    
    # 保存预测结果
    results_df = df_valid.copy()
    results_df['prediction'] = predictions_valid
    results_df['raw_output'] = [raw_outputs[i] for i in range(len(raw_outputs)) if valid_mask[i]]
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # 保存汇总
    summary = {
        'dataset': dataset_name,
        'shot_type': shot_type,
        'n_examples': n_examples if shot_type == 'few-shot' else 0,
        'test_size': len(df_valid),
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存至: {output_dir}")
    print(f"{'='*70}\n")
    
    return summary

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepSeek Few-shot情感分类')
    parser.add_argument('--dataset', type=str, default='chnsenticorp',
                        choices=['chnsenticorp', 'waimai10k'],
                        help='数据集名称')
    parser.add_argument('--shot-type', type=str, default='zero-shot',
                        choices=['zero-shot', 'few-shot'],
                        help='Shot类型')
    parser.add_argument('--n-examples', type=int, default=5,
                        help='Few-shot示例数量')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='测试样本数量')
    
    args = parser.parse_args()
    
    try:
        summary = evaluate_deepseek(
            args.dataset,
            args.shot_type,
            args.n_examples,
            args.sample_size
        )
        
        if summary:
            print("🎉 评估完成!")
    except Exception as e:
        print(f"\n❌ 评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == '__main__':
    main()
