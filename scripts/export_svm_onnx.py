#!/usr/bin/env python3
"""
导出SVM/Logistic模型为ONNX格式，并测试推理速度/精度
"""
import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

MODEL_PATH = 'results/weak_supervision/svm_model.joblib'
VEC_PATH = 'results/weak_supervision/tfidf_vectorizer.joblib'
ONNX_PATH = 'results/weak_supervision/svm_model.onnx'

# 加载模型和向量化器
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# 构造ONNX输入shape
n_features = len(vectorizer.get_feature_names_out())
initial_type = [('input', FloatTensorType([None, n_features]))]

# 转换为ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open(ONNX_PATH, 'wb') as f:
    f.write(onnx_model.SerializeToString())
print(f'✓ 已导出ONNX模型: {ONNX_PATH}')

# 测试推理
texts = ["很好吃，速度快", "太差了，送餐慢"]
X = vectorizer.transform(texts).toarray().astype(np.float32)
sess = ort.InferenceSession(ONNX_PATH)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
output = sess.run([output_name], {input_name: X})[0]
if output.ndim == 1:
    # 只输出类别
    labels = output.astype(int)
    print("ONNX输出为类别:")
    for t, l in zip(texts, labels):
        print(f"文本: {t} | 预测类别: {l}")
else:
    # 输出概率分布
    probs = output[:, 1]
    labels = (probs > 0.5).astype(int)
    for t, l, p in zip(texts, labels, probs):
        print(f"文本: {t} | 预测: {l} | 概率: {p:.4f}")
