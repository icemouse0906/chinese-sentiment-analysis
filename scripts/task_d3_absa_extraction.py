"""
Task D3: ABSA三元组提取
使用DeepSeek实现方面级情感分析（Aspect-Based Sentiment Analysis）
提取（方面词，观点词，情感极性）三元组
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

class ABSAExtractor:
    """ABSA三元组提取器"""
    
    def __init__(self, api_key=None):
        """初始化"""
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
    
    def extract_triplets(self, text, domain='hotel'):
        """
        提取ABSA三元组
        
        Args:
            text: 评论文本
            domain: 领域（hotel/food/ecommerce）
        
        Returns:
            triplets: [{"aspect": "服务", "opinion": "态度好", "sentiment": "正面"}, ...]
        """
        # 根据领域定制提示词
        domain_aspects = {
            'hotel': '服务、环境、位置、设施、价格、卫生',
            'food': '口味、配送、包装、价格、分量、卫生',
            'ecommerce': '质量、价格、物流、包装、外观、功能'
        }
        
        system_prompt = f"""你是一个专业的方面级情感分析（ABSA）专家。请从评论中提取（方面词，观点词，情感极性）三元组。

**任务说明：**
- 方面词（Aspect）：评论关注的具体方面，如{domain_aspects.get(domain, '服务、质量、价格等')}
- 观点词（Opinion）：描述方面的具体表达，如"很好"、"太差"、"一般"等
- 情感极性（Sentiment）：正面/负面/中性

**输出格式：**
严格按照JSON数组格式输出，例如：
[
  {{"aspect": "服务", "opinion": "态度很好", "sentiment": "正面"}},
  {{"aspect": "价格", "opinion": "偏贵", "sentiment": "负面"}}
]

**注意事项：**
1. 如果没有明确的方面词，使用"整体"作为方面词
2. 如果同一方面有多个观点，分别提取
3. 只输出JSON，不要有任何额外解释
"""
        
        user_prompt = f"请分析以下评论的方面级情感：\n\n{text}"
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            result = response.choices[0].message.content.strip()
            
            # 清理markdown代码块标记
            if result.startswith('```'):
                result = result.split('```')[1]
                if result.startswith('json'):
                    result = result[4:]
                result = result.strip()
            
            # 解析JSON
            try:
                triplets = json.loads(result)
                
                # 验证格式
                if not isinstance(triplets, list):
                    return [], result
                
                # 标准化格式
                normalized = []
                for t in triplets:
                    if isinstance(t, dict) and 'aspect' in t and 'sentiment' in t:
                        normalized.append({
                            'aspect': t.get('aspect', ''),
                            'opinion': t.get('opinion', ''),
                            'sentiment': t.get('sentiment', '')
                        })
                
                return normalized, result
            
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON解析失败: {str(e)}")
                print(f"     原始输出: {result[:200]}")
                return [], result
        
        except Exception as e:
            print(f"  ⚠️  提取失败: {str(e)}")
            return [], str(e)
    
    def extract_batch(self, texts, domain='hotel', max_samples=None):
        """批量提取"""
        print(f"\n🔮 批量ABSA三元组提取...")
        print(f"   领域: {domain}")
        print(f"   样本数: {len(texts) if max_samples is None else min(len(texts), max_samples)}")
        
        if max_samples is not None:
            texts = texts[:max_samples]
        
        results = []
        
        for text in tqdm(texts, desc="   提取进度"):
            triplets, raw_output = self.extract_triplets(text, domain)
            
            results.append({
                'text': text,
                'triplets': triplets,
                'n_triplets': len(triplets),
                'raw_output': raw_output
            })
            
            # 限速
            time.sleep(0.8)
        
        return pd.DataFrame(results)

def analyze_absa_distribution(results_df):
    """分析ABSA三元组的分布"""
    print(f"\n📊 ABSA分布统计...")
    
    # 展开所有三元组
    all_triplets = []
    for _, row in results_df.iterrows():
        if isinstance(row['triplets'], list):
            for t in row['triplets']:
                all_triplets.append(t)
    
    if len(all_triplets) == 0:
        print("   ⚠️  没有提取到有效三元组")
        return None
    
    triplets_df = pd.DataFrame(all_triplets)
    
    # 方面词分布
    print(f"\n   方面词分布（Top 10）:")
    aspect_counts = triplets_df['aspect'].value_counts().head(10)
    for aspect, count in aspect_counts.items():
        print(f"     - {aspect}: {count}")
    
    # 情感极性分布
    print(f"\n   情感极性分布:")
    sentiment_counts = triplets_df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"     - {sentiment}: {count} ({count/len(triplets_df)*100:.1f}%)")
    
    # 方面-情感交叉表
    print(f"\n   方面-情感交叉表（Top 5方面）:")
    top_aspects = triplets_df['aspect'].value_counts().head(5).index
    crosstab = pd.crosstab(
        triplets_df[triplets_df['aspect'].isin(top_aspects)]['aspect'],
        triplets_df[triplets_df['aspect'].isin(top_aspects)]['sentiment']
    )
    print(crosstab.to_string())
    
    return {
        'total_triplets': len(all_triplets),
        'unique_aspects': triplets_df['aspect'].nunique(),
        'aspect_distribution': aspect_counts.to_dict(),
        'sentiment_distribution': sentiment_counts.to_dict(),
        'crosstab': crosstab.to_dict()
    }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ABSA三元组提取')
    parser.add_argument('--dataset', type=str, default='chnsenticorp',
                        choices=['chnsenticorp', 'waimai10k', 'ecommerce'],
                        help='数据集名称')
    parser.add_argument('--domain', type=str, default='hotel',
                        choices=['hotel', 'food', 'ecommerce'],
                        help='领域类型')
    parser.add_argument('--sample-size', type=int, default=50,
                        help='分析样本数量')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"ABSA三元组提取".center(70))
    print(f"数据集: {args.dataset} | 领域: {args.domain}".center(70))
    print(f"{'='*70}")
    
    try:
        # 加载数据
        base_dir = Path(__file__).parent.parent
        
        if args.dataset == 'chnsenticorp':
            data_path = base_dir / 'NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv'
        elif args.dataset == 'waimai10k':
            data_path = base_dir / 'NLP数据集/外卖评论数据/waimai_10k.csv'
        elif args.dataset == 'ecommerce':
            data_path = base_dir / 'NLP数据集/电商评论数据/online_shopping_10_cats.csv'
        else:
            raise ValueError(f"未知数据集: {args.dataset}")
        
        print(f"\n📊 加载数据...")
        df = pd.read_csv(data_path)
        
        # 随机采样
        sample_df = df.sample(min(len(df), args.sample_size), random_state=42)
        texts = sample_df['review'].tolist()
        
        print(f"   样本数: {len(texts)}")
        
        # 初始化提取器
        print(f"\n🚀 初始化ABSA提取器...")
        extractor = ABSAExtractor()
        print("   ✅ 连接成功")
        
        # 批量提取
        results_df = extractor.extract_batch(texts, args.domain, args.sample_size)
        
        # 分析分布
        stats = analyze_absa_distribution(results_df)
        
        # 保存结果
        output_dir = Path(__file__).parent.parent / 'results' / 'absa' / args.dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        results_path = output_dir / 'absa_triplets.csv'
        
        # 展开三元组为单独的行（便于分析）
        expanded_rows = []
        for idx, row in results_df.iterrows():
            if isinstance(row['triplets'], list) and len(row['triplets']) > 0:
                for t in row['triplets']:
                    expanded_rows.append({
                        'text': row['text'],
                        'aspect': t.get('aspect', ''),
                        'opinion': t.get('opinion', ''),
                        'sentiment': t.get('sentiment', '')
                    })
            else:
                expanded_rows.append({
                    'text': row['text'],
                    'aspect': '',
                    'opinion': '',
                    'sentiment': ''
                })
        
        expanded_df = pd.DataFrame(expanded_rows)
        expanded_df.to_csv(results_path, index=False)
        
        # 保存原始输出
        raw_path = output_dir / 'absa_raw_outputs.json'
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(results_df.to_dict('records'), f, indent=2, ensure_ascii=False)
        
        # 保存统计信息
        if stats:
            stats_path = output_dir / 'absa_statistics.json'
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 生成可读报告
        report_path = output_dir / 'absa_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# ABSA三元组提取报告\n\n")
            f.write(f"**数据集**: {args.dataset}\n")
            f.write(f"**领域**: {args.domain}\n")
            f.write(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**分析样本数**: {len(results_df)}\n")
            
            if stats:
                f.write(f"**提取三元组总数**: {stats['total_triplets']}\n")
                f.write(f"**唯一方面词数**: {stats['unique_aspects']}\n\n")
            
            f.write("---\n\n## 示例结果\n\n")
            
            # 展示前5个示例
            for idx, row in results_df.head(5).iterrows():
                f.write(f"### 示例 {idx + 1}\n\n")
                f.write(f"**评论**: {row['text']}\n\n")
                f.write(f"**提取的三元组** ({row['n_triplets']}个):\n\n")
                
                if isinstance(row['triplets'], list) and len(row['triplets']) > 0:
                    for t in row['triplets']:
                        f.write(f"- 方面: **{t.get('aspect', '')}** | 观点: _{t.get('opinion', '')}_ | 情感: **{t.get('sentiment', '')}**\n")
                else:
                    f.write("_未提取到三元组_\n")
                
                f.write("\n---\n\n")
        
        print(f"\n✅ 提取完成！")
        print(f"   结果已保存至: {output_dir}")
        print(f"   - 三元组CSV: {results_path}")
        print(f"   - 统计信息: {output_dir / 'absa_statistics.json'}")
        print(f"   - 报告文件: {report_path}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ 提取失败: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == '__main__':
    main()
