import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import json

def main():
    parser = argparse.ArgumentParser(description='中文句向量生成与检索')
    parser.add_argument('--input', type=str, required=True, help='输入CSV，需含text列')
    parser.add_argument('--output', type=str, default='output/sentence_embeddings.npy', help='输出npy文件')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-base-zh', help='句向量模型')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    model = SentenceTransformer(args.model_name)
    embeddings = model.encode(df['text'].tolist(), batch_size=32, show_progress_bar=True)
    np.save(args.output, embeddings)
    print(f'句向量已保存到 {args.output}')
    # Top-5相似句检索示例
    idx = 0
    query_emb = embeddings[idx]
    sims = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    top5 = np.argsort(sims)[-6:-1][::-1]
    print('Top-5相似句：')
    for i in top5:
        print(df.iloc[i]['text'])
    with open(args.output.replace('.npy','.top5.json'), 'w', encoding='utf-8') as f:
        json.dump([df.iloc[i]['text'] for i in top5], f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
