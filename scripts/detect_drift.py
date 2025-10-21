"""
数据漂移检测脚本
使用Evidently库检测训练集与生产数据的分布差异
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently import ColumnMapping
except ImportError:
    print("⚠️  Evidently库未安装，跳过漂移检测")
    print("   安装命令: pip install evidently")
    sys.exit(0)

def load_reference_data(dataset_name):
    """加载参考数据（训练集）"""
    base_dir = Path(__file__).parent.parent
    
    if dataset_name == 'chnsenticorp':
        data_path = base_dir / 'NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv'
    elif dataset_name == 'waimai10k':
        data_path = base_dir / 'NLP数据集/外卖评论数据/waimai_10k.csv'
    else:
        return None
    
    if not data_path.exists():
        return None
    
    df = pd.read_csv(data_path)
    # 模拟：取前80%作为参考数据
    reference = df.iloc[:int(len(df)*0.8)]
    return reference

def load_current_data(dataset_name):
    """加载当前数据（生产/测试数据）"""
    base_dir = Path(__file__).parent.parent
    
    if dataset_name == 'chnsenticorp':
        data_path = base_dir / 'NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv'
    elif dataset_name == 'waimai10k':
        data_path = base_dir / 'NLP数据集/外卖评论数据/waimai_10k.csv'
    else:
        return None
    
    if not data_path.exists():
        return None
    
    df = pd.read_csv(data_path)
    # 模拟：取后20%作为当前数据
    current = df.iloc[int(len(df)*0.8):]
    return current

def detect_drift(dataset_name):
    """检测数据漂移"""
    print(f"\n{'='*60}")
    print(f"检测数据集漂移: {dataset_name}".center(60))
    print(f"{'='*60}")
    
    # 加载数据
    reference = load_reference_data(dataset_name)
    current = load_current_data(dataset_name)
    
    if reference is None or current is None:
        print(f"⚠️  数据集 {dataset_name} 不存在，跳过")
        return None
    
    print(f"参考数据: {len(reference)} 行")
    print(f"当前数据: {len(current)} 行")
    
    # 添加数值特征（文本长度）
    reference['text_length'] = reference['review'].str.len()
    current['text_length'] = current['review'].str.len()
    
    # 配置列映射
    column_mapping = ColumnMapping(
        target='label',
        numerical_features=['text_length'],
        text_features=['review']
    )
    
    # 生成漂移报告
    print("生成Evidently报告...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])
    
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping
    )
    
    # 保存报告
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / f'data_drift_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
    report.save_html(str(report_path))
    
    print(f"✅ 漂移检测完成")
    print(f"   报告已保存: {report_path}")
    
    # 获取漂移统计
    report_dict = report.as_dict()
    drift_share = None
    
    try:
        for metric in report_dict.get('metrics', []):
            if metric.get('metric') == 'DatasetDriftMetric':
                drift_share = metric['result'].get('drift_share', 0)
                n_drifted = metric['result'].get('number_of_drifted_columns', 0)
                
                print(f"\n漂移统计:")
                print(f"  漂移特征比例: {drift_share:.2%}")
                print(f"  漂移特征数量: {n_drifted}")
                
                if drift_share > 0.3:
                    print(f"  ⚠️  警告：超过30%的特征发生漂移！")
                    return 'WARNING'
                elif drift_share > 0:
                    print(f"  ℹ️  提示：检测到轻微漂移")
                    return 'INFO'
                else:
                    print(f"  ✅ 未检测到显著漂移")
                    return 'OK'
    except Exception as e:
        print(f"⚠️  解析漂移统计失败: {str(e)}")
    
    return 'OK'

def main():
    """主函数"""
    print("=" * 60)
    print("数据漂移检测开始".center(60))
    print("=" * 60)
    
    datasets = ['chnsenticorp', 'waimai10k']
    results = {}
    
    for dataset in datasets:
        try:
            status = detect_drift(dataset)
            results[dataset] = status
        except Exception as e:
            print(f"❌ 检测失败: {str(e)}")
            import traceback
            traceback.print_exc()
            results[dataset] = 'ERROR'
    
    print(f"\n{'='*60}")
    print("漂移检测完成".center(60))
    print(f"{'='*60}")
    
    for dataset, status in results.items():
        emoji = '✅' if status == 'OK' else '⚠️' if status == 'WARNING' else '❌'
        print(f"{emoji} {dataset}: {status}")
    
    # 如果有警告或错误，退出码为1（CI/CD中可设置continue-on-error）
    if 'WARNING' in results.values() or 'ERROR' in results.values():
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
