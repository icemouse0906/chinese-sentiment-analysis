import os
import pandas as pd

WORKDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPATH = os.path.join(WORKDIR, 'output', 'labels_ecommerce.csv')
OUTDIR = os.path.join(WORKDIR, 'output')
HEAD_OUT = os.path.join(OUTDIR, 'labels_ecommerce_head.csv')
COUNT_OUT = os.path.join(OUTDIR, 'labels_ecommerce_label_counts.csv')

os.makedirs(OUTDIR, exist_ok=True)

def read_maybe(path):
    # try utf-8 then latin1; use default C engine for better performance
    try:
        return pd.read_csv(path, encoding='utf-8')
    except Exception as e:
        print('utf-8 read failed:', e)
    try:
        return pd.read_csv(path, encoding='latin1')
    except Exception as e:
        print('latin1 read failed:', e)
        raise

if __name__ == '__main__':
    print('Reading', INPATH)
    df = read_maybe(INPATH)
    # write head
    head = df.head(50)
    head.to_csv(HEAD_OUT, index=False)
    # compute label counts (if column exists)
    if 'sentiment_label' in df.columns:
        counts = df['sentiment_label'].value_counts(dropna=False).rename_axis('sentiment_label').reset_index(name='counts')
    else:
        # try to infer label column name
        potential = [c for c in df.columns if 'label' in c.lower() or 'sentiment' in c.lower()]
        if potential:
            col = potential[0]
            counts = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name='counts')
        else:
            counts = pd.DataFrame({'note': ['no_label_column_found']})
    counts.to_csv(COUNT_OUT, index=False)
    print('Wrote', HEAD_OUT)
    print('Wrote', COUNT_OUT)
