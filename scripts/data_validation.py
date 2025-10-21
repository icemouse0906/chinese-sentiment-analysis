"""
数据验证脚本 - 用于MLOps流水线
检查数据完整性、Schema、统计特性
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

def validate_schema(df, dataset_name, expected_columns):
    """验证DataFrame的Schema"""
    errors = []
    
    # 检查列名
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"缺少必需列: {missing_cols}")
    
    # 检查空值
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        errors.append(f"存在空值: {null_cols.to_dict()}")
    
    # 检查数据类型
    if 'label' in df.columns:
        unique_labels = df['label'].unique()
        if not set(unique_labels).issubset({0, 1}):
            errors.append(f"标签值异常: {unique_labels}")
    
    return errors

def validate_statistics(df, dataset_name):
    """验证统计特性"""
    warnings = []
    
    # 文本长度检查
    if 'review' in df.columns:
        text_lengths = df['review'].str.len()
        if text_lengths.min() < 5:
            warnings.append(f"存在过短文本(< 5字符): {(text_lengths < 5).sum()}条")
        if text_lengths.max() > 1000:
            warnings.append(f"存在超长文本(> 1000字符): {(text_lengths > 1000).sum()}条")
    
    # 类别平衡检查
    if 'label' in df.columns:
        label_dist = df['label'].value_counts(normalize=True)
        imbalance_ratio = label_dist.max() / label_dist.min()
        if imbalance_ratio > 3:
            warnings.append(f"类别不平衡严重: {label_dist.to_dict()}")
    
    return warnings

def main():
    """主函数"""
    print("=" * 60)
    print("数据验证开始".center(60))
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    data_configs = {
        'chnsenticorp': {
            'path': base_dir / 'NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv',
            'columns': ['review', 'label']
        },
        'waimai10k': {
            'path': base_dir / 'NLP数据集/外卖评论数据/waimai_10k.csv',
            'columns': ['review', 'label']
        },
        'ecommerce': {
            'path': base_dir / 'NLP数据集/电商评论数据/online_shopping_10_cats.csv',
            'columns': ['review', 'cat', 'label']
        }
    }
    
    validation_results = {}
    all_passed = True
    
    for dataset_name, config in data_configs.items():
        print(f"\n{'='*60}")
        print(f"验证数据集: {dataset_name}".center(60))
        print(f"{'='*60}")
        
        try:
            # 读取数据
            if not config['path'].exists():
                print(f"⚠️  数据文件不存在: {config['path']}")
                validation_results[dataset_name] = {
                    'status': 'SKIPPED',
                    'reason': '文件不存在'
                }
                continue
            
            df = pd.read_csv(config['path'])
            print(f"✓ 数据加载成功: {len(df)} 行")
            
            # Schema验证
            schema_errors = validate_schema(df, dataset_name, config['columns'])
            if schema_errors:
                print(f"❌ Schema验证失败:")
                for error in schema_errors:
                    print(f"   - {error}")
                all_passed = False
            else:
                print(f"✓ Schema验证通过")
            
            # 统计特性验证
            stat_warnings = validate_statistics(df, dataset_name)
            if stat_warnings:
                print(f"⚠️  统计特性警告:")
                for warning in stat_warnings:
                    print(f"   - {warning}")
            else:
                print(f"✓ 统计特性正常")
            
            # 记录结果
            validation_results[dataset_name] = {
                'status': 'PASSED' if not schema_errors else 'FAILED',
                'rows': len(df),
                'errors': schema_errors,
                'warnings': stat_warnings,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ 验证过程出错: {str(e)}")
            validation_results[dataset_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
            all_passed = False
    
    # 保存验证报告
    output_dir = base_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / f'data_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("验证完成".center(60))
    print(f"{'='*60}")
    print(f"报告已保存: {report_path}")
    
    if all_passed:
        print("✅ 所有数据集验证通过")
        sys.exit(0)
    else:
        print("❌ 部分数据集验证失败")
        sys.exit(1)

if __name__ == '__main__':
    main()
