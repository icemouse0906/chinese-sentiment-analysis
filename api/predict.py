from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

app = FastAPI()

class Item(BaseModel):
    text: str

# 加载模型（示例：MacBERT）
MODEL_NAME = 'hfl/chinese-macbert-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

@app.post('/predict')
def predict(item: Item):
    inputs = tokenizer(item.text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        label = int(probs.argmax())
        conf = float(probs[label])
    return {'label': label, 'confidence': conf}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
