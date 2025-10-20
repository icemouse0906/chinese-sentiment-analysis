# coding: utf-8
"""
对正负两类分别做 LDA 主题分析（使用 gensim）
- 输入：output/labels_*.csv（需先运行 03_label_and_model.py）
- 输出：output/lda_topics_{file}_{pos|neg}.txt，保存每个主题关键词
"""
from pathlib import Path
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
DATA = OUT
OUT.mkdir(exist_ok=True)

FILES = ['hotel', 'ecommerce', 'waimai']

# If a dataset is large (e.g. ecommerce), cap the number of documents used for LDA
SAMPLE_CAP = {
    'ecommerce': 20000,
    'hotel': 5000,
    'waimai': 5000
}


def run_lda_on_texts(texts, out_prefix, num_topics=6):
    # texts: list of token lists (list of lists of str)
    if not texts:
        Path(f'{out_prefix}.txt').write_text('No texts to analyze', encoding='utf-8')
        return
    dct = Dictionary(texts)
    corpus = [dct.doc2bow(t) for t in texts]
    lda = LdaModel(corpus=corpus, id2word=dct, num_topics=num_topics, random_state=42, passes=10)
    topics = lda.print_topics(num_words=10)
    # write readable UTF-8 topics
    Path(f'{out_prefix}.txt').write_text('\n'.join([str(t) for t in topics]), encoding='utf-8')


if __name__ == '__main__':
    for name in FILES:
        # Prefer processed CSV in data/ if available (these should contain readable tokens)
        proc_path = ROOT / 'data' / f'processed_{name}.csv'
        label_path = DATA / f'labels_{name}.csv'

        if proc_path.exists():
            df = pd.read_csv(proc_path)
            source = proc_path
        elif label_path.exists():
            df = pd.read_csv(label_path)
            source = label_path
        else:
            print('Missing both processed and label files for', name)
            continue

        # ensure tokens column is available and readable
        if 'tokens' in df.columns:
            # tokens may be stored as string repr of list
            if df['tokens'].dtype == object:
                try:
                    df['tokens'] = df['tokens'].apply(eval)
                except Exception:
                    # fallback: split tokens_join if available
                    if 'tokens_join' in df.columns:
                        df['tokens'] = df['tokens_join'].fillna('').apply(lambda s: s.split())
                    else:
                        df['tokens'] = df.iloc[:,0].fillna('').apply(lambda s: str(s).split())
        elif 'tokens_join' in df.columns:
            df['tokens'] = df['tokens_join'].fillna('').apply(lambda s: str(s).split())
        else:
            print('No tokens/tokens_join found in', source)
            continue

        # determine labels: prefer sentiment_label column; otherwise compute with SnowNLP
        if 'sentiment_label' in df.columns:
            labels = df['sentiment_label']
        else:
            try:
                from snownlp import SnowNLP
                scores = df['tokens_join'].fillna('').apply(lambda s: SnowNLP(str(s)).sentiments if str(s).strip() else 0.5)
                labels = scores.apply(lambda x: 1 if x>=0.5 else 0)
            except Exception:
                # fallback: all negative
                labels = pd.Series([0]*len(df))

        df['sentiment_label'] = labels

        for label, grp in df.groupby('sentiment_label'):
            texts = list(grp['tokens'])
            # sample if too large
            cap = SAMPLE_CAP.get(name, None)
            if cap and len(texts) > cap:
                import random
                random.seed(42)
                texts = random.sample(texts, cap)
            tname = 'pos' if label==1 else 'neg'
            outp = OUT / f'lda_topics_{name}_{tname}'
            run_lda_on_texts(texts, outp)
            print('Wrote', outp)
