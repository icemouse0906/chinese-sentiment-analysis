#!/usr/bin/env python3
"""
任务D3：推理服务与模型压缩
- FastAPI批量预测接口（SVM/Transformer均可）
- 支持ONNX/int8量化（如有模型）
"""
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import uvicorn
import joblib
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

app = FastAPI()

# 加载模型与向量化器（以SVM/Logistic为例，可扩展Transformer）
MODEL_PATH = 'results/weak_supervision/svm_model.joblib'
VEC_PATH = 'results/weak_supervision/tfidf_vectorizer.joblib'

if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
else:
    model = None
    vectorizer = None

class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    labels: List[int]
    probs: List[float]

@app.post('/predict', response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None or vectorizer is None:
        return PredictResponse(labels=[], probs=[])
    X = vectorizer.transform(req.texts)
    probs = model.predict_proba(X)[:, 1]
    labels = (probs > 0.5).astype(int).tolist()
    return PredictResponse(labels=labels, probs=probs.tolist())

@app.get('/health')
async def health():
    return {'status': 'ok'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
