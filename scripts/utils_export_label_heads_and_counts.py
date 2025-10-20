# coding: utf-8
"""
导出 labels_{hotel,waimai}.csv 的前 50 行和 sentiment_label 计数，写到 output/
使用场景：labels_ecommerce 过大时我们已用专用脚本导出 head & counts，这里补 hotel/waimai
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

FILES = ['labels_hotel.csv', 'labels_waimai.csv']

for fn in FILES:
    p = OUT / fn
    if not p.exists():
        print('Missing', p)
        continue
    try:
        df = pd.read_csv(p, encoding='utf-8')
    except Exception:
        df = pd.read_csv(p, encoding='latin1')

    head = df.head(50)
    counts = df['sentiment_label'].value_counts().sort_index().rename_axis('sentiment_label').reset_index(name='counts')

    head.to_csv(OUT / f'{fn.replace(".csv","")}_head.csv', index=False, encoding='utf-8')
    counts.to_csv(OUT / f'{fn.replace(".csv","")}_label_counts.csv', index=False, encoding='utf-8')
    print('Wrote', fn)
