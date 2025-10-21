"""
Task D2: RAG增强误判分析
使用句向量检索 + DeepSeek生成，为误判样本提供解释性分析
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI
import json
from datetime import datetime
from tqdm import tqdm
import time

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("⚠️  需要安装sentence-transformers和faiss-cpu")
    print("   pip install sentence-transformers faiss-cpu")
    exit(1)

class RAGErrorAnalyzer:
    """RAG增强的误判分析器"""
    
    def __init__(self, api_key=None, embedding_model='BAAI/bge-large-zh-v1.5'):
        """
        初始化RAG分析器
        
        Args:
            api_key: DeepSeek API密钥
            embedding_model: 句向量模型
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        
        # 加载句向量模型
        print(f"📥 加载句向量模型: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        print("   ✅ 模型加载完成")
        
        self.error_db = None
        self.error_samples = None
        self.index = None
    
    def build_error_database(self, error_samples_df):
        """
        构建误判样本的向量数据库
        
        Args:
            error_samples_df: 误判样本DataFrame (包含text, true_label, pred_label列)
        """
        print(f"\n🔨 构建误判样本向量数据库...")
        print(f"   样本数量: {len(error_samples_df)}")
        
        self.error_samples = error_samples_df.copy()
        
        # 生成句向量
        print("   生成句向量...")
        texts = error_samples_df['text'].tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # 构建FAISS索引
        print("   构建FAISS索引...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 内积（余弦相似度）
        
        # 归一化（余弦相似度需要）
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.error_db = embeddings
        print(f"   ✅ 数据库构建完成 (维度: {dimension})")
    
    def retrieve_similar_errors(self, query_text, top_k=3):
        """
        检索相似的误判样本
        
        Args:
            query_text: 查询文本
            top_k: 返回top-k个相似样本
        """
        if self.index is None:
            raise ValueError("请先调用build_error_database()构建数据库")
        
        # 查询向量
        query_embedding = self.embedder.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # 检索
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # 返回结果
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < len(self.error_samples):
                sample = self.error_samples.iloc[idx]
                results.append({
                    'text': sample['text'],
                    'true_label': sample['true_label'],
                    'pred_label': sample['pred_label'],
                    'similarity': float(sim),
                    'index': int(idx)
                })
        
        return results
    
    def generate_error_analysis(self, query_sample, similar_cases):
        """
        使用DeepSeek生成误判分析报告
        
        Args:
            query_sample: 当前误判样本 (dict with text, true_label, pred_label)
            similar_cases: 相似误判样本列表
        """
        # 构造提示词
        system_prompt = """你是一个专业的情感分析模型诊断专家。请根据给定的误判样本和相似案例，分析模型为什么会犯错。

请从以下角度分析：
1. **文本特征**：句子结构、转折词、情感词等
2. **情感表达**：隐式情感、反讽、客套话等
3. **上下文依赖**：是否需要领域知识或常识推理
4. **相似案例模式**：从相似错误中总结共性

输出格式：
{
  "error_reason": "主要错误原因的简短描述",
  "detailed_analysis": "详细分析（100-200字）",
  "suggested_fix": "改进建议"
}"""
        
        # 构造用户提示
        true_label_text = "正面" if query_sample['true_label'] == 1 else "负面"
        pred_label_text = "正面" if query_sample['pred_label'] == 1 else "负面"
        
        user_prompt = f"""**当前误判样本：**
文本：{query_sample['text']}
真实标签：{true_label_text}
预测标签：{pred_label_text}

**相似误判案例（共{len(similar_cases)}条）：**
"""
        
        for i, case in enumerate(similar_cases, 1):
            case_true = "正面" if case['true_label'] == 1 else "负面"
            case_pred = "正面" if case['pred_label'] == 1 else "负面"
            user_prompt += f"\n{i}. 文本：{case['text'][:100]}...\n   真实：{case_true} | 预测：{case_pred} | 相似度：{case['similarity']:.3f}\n"
        
        user_prompt += "\n请分析这个误判案例的原因。"
        
        # 调用DeepSeek
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            result = response.choices[0].message.content.strip()
            
            # 尝试解析JSON
            try:
                analysis = json.loads(result)
            except:
                # 如果不是JSON，包装成字典
                analysis = {
                    "error_reason": "未知",
                    "detailed_analysis": result,
                    "suggested_fix": "需要进一步分析"
                }
            
            return analysis
        
        except Exception as e:
            print(f"  ⚠️  分析生成失败: {str(e)}")
            return {
                "error_reason": "API调用失败",
                "detailed_analysis": str(e),
                "suggested_fix": "检查API连接"
            }
    
    def analyze_batch_errors(self, error_samples_df, top_k=3, max_samples=50):
        """
        批量分析误判样本
        
        Args:
            error_samples_df: 误判样本DataFrame
            top_k: 检索相似样本数量
            max_samples: 最大分析样本数（控制成本）
        """
        print(f"\n🔍 批量误判分析...")
        print(f"   总误判样本: {len(error_samples_df)}")
        print(f"   分析样本数: {min(len(error_samples_df), max_samples)}")
        
        # 先构建数据库
        self.build_error_database(error_samples_df)
        
        # 随机选择样本（或选择置信度最低的）
        if len(error_samples_df) > max_samples:
            samples_to_analyze = error_samples_df.sample(max_samples, random_state=42)
        else:
            samples_to_analyze = error_samples_df
        
        analyses = []
        
        for idx, row in tqdm(samples_to_analyze.iterrows(), total=len(samples_to_analyze), desc="   分析进度"):
            query_sample = {
                'text': row['text'],
                'true_label': row['true_label'],
                'pred_label': row['pred_label']
            }
            
            # 检索相似案例
            similar_cases = self.retrieve_similar_errors(row['text'], top_k)
            
            # 生成分析
            analysis = self.generate_error_analysis(query_sample, similar_cases)
            
            # 记录结果
            analyses.append({
                'index': idx,
                'text': row['text'],
                'true_label': row['true_label'],
                'pred_label': row['pred_label'],
                'error_reason': analysis.get('error_reason', ''),
                'detailed_analysis': analysis.get('detailed_analysis', ''),
                'suggested_fix': analysis.get('suggested_fix', ''),
                'similar_cases': similar_cases
            })
            
            # 限速
            time.sleep(1)
        
        return pd.DataFrame(analyses)

def load_error_samples(dataset_name, model_type='svm'):
    """加载模型的误判样本"""
    base_dir = Path(__file__).parent.parent
    
    # 加载测试数据和预测结果
    if dataset_name == 'chnsenticorp':
        data_path = base_dir / 'NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv'
    elif dataset_name == 'waimai10k':
        data_path = base_dir / 'NLP数据集/外卖评论数据/waimai_10k.csv'
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 简单模拟：随机生成一些误判样本（实际应从模型预测结果中读取）
    # 在实际应用中，应该从 results/{dataset}/{model}/predictions.csv 读取
    
    # 这里我们从跨域实验的误判样本中读取
    error_samples_path = base_dir / 'results' / 'cross_domain' / 'error_samples.csv'
    
    if error_samples_path.exists():
        error_df = pd.read_csv(error_samples_path)
        # 过滤特定数据集的误判
        if dataset_name == 'chnsenticorp':
            error_df = error_df[error_df['transfer'].str.contains('hotel|酒店', case=False, na=False)]
        elif dataset_name == 'waimai10k':
            error_df = error_df[error_df['transfer'].str.contains('takeaway|外卖', case=False, na=False)]
        
        # 重命名列
        error_df = error_df.rename(columns={
            'text': 'text',
            'true_label': 'true_label',
            'pred_label': 'pred_label'
        })
        
        return error_df[['text', 'true_label', 'pred_label']].head(100)
    else:
        # 如果没有误判样本，随机生成一些示例
        print("⚠️  未找到误判样本文件，使用模拟数据")
        sample_errors = df.sample(50, random_state=42).copy()
        # 随机翻转一些标签作为"误判"
        sample_errors['pred_label'] = sample_errors['label'].apply(lambda x: 1 - x if np.random.rand() > 0.7 else x)
        sample_errors = sample_errors[sample_errors['label'] != sample_errors['pred_label']]
        
        return sample_errors.rename(columns={'review': 'text', 'label': 'true_label'})[['text', 'true_label', 'pred_label']]

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG增强误判分析')
    parser.add_argument('--dataset', type=str, default='chnsenticorp',
                        choices=['chnsenticorp', 'waimai10k'],
                        help='数据集名称')
    parser.add_argument('--model', type=str, default='svm',
                        choices=['nb', 'svm'],
                        help='模型类型')
    parser.add_argument('--top-k', type=int, default=3,
                        help='检索相似样本数量')
    parser.add_argument('--max-samples', type=int, default=20,
                        help='最大分析样本数')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"RAG增强误判分析".center(70))
    print(f"数据集: {args.dataset} | 模型: {args.model.upper()}".center(70))
    print(f"{'='*70}")
    
    try:
        # 加载误判样本
        print("\n📊 加载误判样本...")
        error_samples = load_error_samples(args.dataset, args.model)
        print(f"   误判样本数: {len(error_samples)}")
        
        if len(error_samples) == 0:
            print("❌ 没有误判样本可分析")
            return
        
        # 初始化分析器
        print("\n🚀 初始化RAG分析器...")
        analyzer = RAGErrorAnalyzer()
        
        # 批量分析
        analysis_df = analyzer.analyze_batch_errors(
            error_samples,
            top_k=args.top_k,
            max_samples=args.max_samples
        )
        
        # 保存结果
        output_dir = Path(__file__).parent.parent / 'results' / 'rag_analysis' / args.dataset / args.model
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_path = output_dir / 'error_analysis.csv'
        analysis_df.to_csv(analysis_path, index=False)
        
        # 生成可读报告
        report_path = output_dir / 'error_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# RAG增强误判分析报告\n\n")
            f.write(f"**数据集**: {args.dataset}\n")
            f.write(f"**模型**: {args.model.upper()}\n")
            f.write(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**分析样本数**: {len(analysis_df)}\n\n")
            f.write("---\n\n")
            
            for idx, row in analysis_df.iterrows():
                f.write(f"## 案例 {idx + 1}\n\n")
                f.write(f"**文本**: {row['text']}\n\n")
                f.write(f"**真实标签**: {'正面' if row['true_label'] == 1 else '负面'}\n")
                f.write(f"**预测标签**: {'正面' if row['pred_label'] == 1 else '负面'}\n\n")
                f.write(f"### 错误原因\n{row['error_reason']}\n\n")
                f.write(f"### 详细分析\n{row['detailed_analysis']}\n\n")
                f.write(f"### 改进建议\n{row['suggested_fix']}\n\n")
                f.write("---\n\n")
        
        print(f"\n✅ 分析完成！")
        print(f"   结果已保存至: {output_dir}")
        print(f"   - CSV文件: {analysis_path}")
        print(f"   - 报告文件: {report_path}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == '__main__':
    main()
