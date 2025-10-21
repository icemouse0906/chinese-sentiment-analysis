#!/usr/bin/env python3
"""
环境检查脚本
验证所有依赖是否正确安装
"""

import sys
from pathlib import Path

def check_package(package_name, import_name=None):
    """检查Python包是否可导入"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name} - {str(e)}")
        return False

def check_env_var(var_name):
    """检查环境变量是否设置"""
    import os
    value = os.getenv(var_name)
    if value:
        print(f"✅ {var_name} = {value[:10]}...")
        return True
    else:
        print(f"⚠️  {var_name} 未设置")
        return False

def main():
    print("="*60)
    print("环境检查开始".center(60))
    print("="*60)
    
    # 基础依赖
    print("\n📦 基础依赖:")
    basic_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('jieba', 'jieba'),
    ]
    
    basic_ok = all(check_package(name, imp) for name, imp in basic_packages)
    
    # MLOps依赖
    print("\n🔧 MLOps依赖:")
    mlops_packages = [
        ('mlflow', 'mlflow'),
        ('evidently', 'evidently'),
    ]
    
    mlops_ok = all(check_package(name, imp) for name, imp in mlops_packages)
    
    # LLM依赖
    print("\n🤖 LLM相关依赖:")
    llm_packages = [
        ('openai', 'openai'),
        ('sentence-transformers', 'sentence_transformers'),
        ('faiss-cpu', 'faiss'),
    ]
    
    llm_ok = all(check_package(name, imp) for name, imp in llm_packages)
    
    # 环境变量
    print("\n🔑 环境变量:")
    env_ok = check_env_var('DEEPSEEK_API_KEY')
    
    # 数据文件
    print("\n📁 数据文件:")
    base_dir = Path(__file__).parent.parent
    data_files = [
        'NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv',
        'NLP数据集/外卖评论数据/waimai_10k.csv',
        'NLP数据集/电商评论数据/online_shopping_10_cats.csv',
    ]
    
    data_ok = True
    for file_path in data_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (不存在)")
            data_ok = False
    
    # 总结
    print(f"\n{'='*60}")
    print("检查结果".center(60))
    print(f"{'='*60}")
    
    all_ok = basic_ok and mlops_ok and llm_ok and data_ok
    
    if all_ok and env_ok:
        print("✅ 所有检查通过！环境就绪！")
        print("\n📖 快速开始：")
        print("   python scripts/task_d1_deepseek_fewshot.py --help")
        return 0
    else:
        print("⚠️  部分检查未通过")
        
        if not basic_ok:
            print("\n修复基础依赖：")
            print("   pip install pandas numpy scikit-learn jieba")
        
        if not mlops_ok:
            print("\n修复MLOps依赖：")
            print("   pip install mlflow evidently")
        
        if not llm_ok:
            print("\n修复LLM依赖：")
            print("   pip install openai sentence-transformers faiss-cpu")
        
        if not env_ok:
            print("\n配置环境变量：")
            print("   export DEEPSEEK_API_KEY='sk-your-api-key'")
            print("   详见: docs/DEEPSEEK_SETUP.md")
        
        if not data_ok:
            print("\n数据文件说明：")
            print("   请参考README.md中的数据集下载链接")
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
