"""
MLflow实验追踪脚本
集成到模型训练流程，记录超参数、指标、模型artifacts
"""

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import jieba
import json
from datetime import datetime

def load_data(dataset_name):
    """加载数据集"""
    base_dir = Path(__file__).parent.parent
    
    if dataset_name == 'chnsenticorp':
        data_path = base_dir / 'NLP数据集/酒店评论数据/ChnSentiCorp_htl_all.csv'
    elif dataset_name == 'waimai10k':
        data_path = base_dir / 'NLP数据集/外卖评论数据/waimai_10k.csv'
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    df = pd.read_csv(data_path)
    return df

def preprocess_text(text):
    """文本预处理"""
    if pd.isna(text):
        return ""
    tokens = jieba.lcut(str(text))
    return ' '.join(tokens)

def train_model(dataset_name, model_type, experiment_name):
    """训练模型并记录到MLflow"""
    
    # 设置MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{dataset_name}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # 记录参数
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model_type", model_type)
        
        # 加载数据
        print(f"加载数据集: {dataset_name}")
        df = load_data(dataset_name)
        mlflow.log_param("dataset_size", len(df))
        
        # 数据预处理
        print("文本预处理...")
        df['processed'] = df['review'].apply(preprocess_text)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed'], df['label'], 
            test_size=0.2, random_state=42, stratify=df['label']
        )
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # 特征提取
        print("TF-IDF特征提取...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        mlflow.log_param("max_features", 5000)
        
        # 模型训练
        print(f"训练{model_type.upper()}模型...")
        if model_type == 'nb':
            model = MultinomialNB()
            mlflow.log_param("model_class", "MultinomialNB")
        elif model_type == 'svm':
            model = SVC(kernel='linear', probability=True, random_state=42)
            mlflow.log_param("model_class", "SVC")
            mlflow.log_param("kernel", "linear")
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        
        model.fit(X_train_vec, y_train)
        
        # 预测
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # 记录指标
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        
        print(f"\n模型性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  宏F1: {f1_macro:.4f}")
        print(f"  加权F1: {f1_weighted:.4f}")
        
        # 分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 保存模型和artifacts
        output_dir = Path(__file__).parent.parent / 'results' / dataset_name / model_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存分类报告
        report_df = pd.DataFrame(report).T
        report_path = output_dir / 'classification_report.csv'
        report_df.to_csv(report_path)
        mlflow.log_artifact(str(report_path))
        
        # 保存模型
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"{dataset_name}_{model_type}"
        )
        
        # 保存vectorizer
        import pickle
        vectorizer_path = output_dir / 'vectorizer.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact(str(vectorizer_path))
        
        # 记录标签
        mlflow.set_tags({
            "dataset": dataset_name,
            "model_type": model_type,
            "stage": "production" if f1_macro > 0.65 else "staging"
        })
        
        print(f"\n✅ MLflow运行完成")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'run_id': mlflow.active_run().info.run_id
        }

def main():
    parser = argparse.ArgumentParser(description='MLflow实验追踪')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['chnsenticorp', 'waimai10k'],
                        help='数据集名称')
    parser.add_argument('--model', type=str, required=True,
                        choices=['nb', 'svm'],
                        help='模型类型')
    parser.add_argument('--experiment-name', type=str, default='sentiment-analysis',
                        help='MLflow实验名称')
    
    args = parser.parse_args()
    
    try:
        results = train_model(args.dataset, args.model, args.experiment_name)
        print(f"\n{'='*60}")
        print("训练完成".center(60))
        print(f"{'='*60}")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == '__main__':
    main()
