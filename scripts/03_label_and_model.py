# coding: utf-8
"""
生成情感伪标签并训练基线模型
- 使用 SnowNLP 对无标签数据打分（需 pip install snownlp）
- 以默认阈值 0.5 划分正/负
- 使用 TF-IDF + 朴素贝叶斯 与 SVM 做基线，输出报告（classification_report）

运行：
python scripts/03_label_and_model.py

输出：
- output/labels_*.csv
- output/classification_report_*.txt
"""
from pathlib import Path
import pandas as pd
from snownlp import SnowNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.utils import resample
import warnings

ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = ROOT / 'data'
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

FILES = {
    'hotel': DATA_OUT / 'processed_hotel.csv',
    'ecommerce': DATA_OUT / 'processed_ecommerce.csv',
    'waimai': DATA_OUT / 'processed_waimai.csv'
}


def score_and_label(df, text_col='tokens_join', threshold=0.5):
    scores = []
    for t in df[text_col].fillna(''):
        # ensure we have a string
        t_str = str(t)
        if not t_str.strip():
            # empty text -> neutral score
            scores.append(0.5)
            continue
        try:
            s = SnowNLP(t_str).sentiments
        except Exception:
            # if SnowNLP fails (e.g. internal BM25 on empty docs), use neutral score
            s = 0.5
        scores.append(s)
    df['sentiment_score'] = scores
    df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 1 if x>=threshold else 0)
    return df


if __name__ == '__main__':
    for name, path in FILES.items():
        if not path.exists():
            print('Missing', path)
            continue
        # robust CSV read with fallbacks
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except Exception:
            try:
                df = pd.read_csv(path, encoding='latin1')
            except Exception:
                df = pd.read_csv(path, engine='python', on_bad_lines='skip')

        df = score_and_label(df, text_col='tokens_join')
        # ensure we write out with utf-8 so downstream exports are readable
        df.to_csv(OUT / f'labels_{name}.csv', index=False, encoding='utf-8')
        print('Wrote labels to', OUT / f'labels_{name}.csv')

        # Prepare train/test split on the original dataframe (so we can resample easily)
        y = df['sentiment_label']
        # if any class has <2 examples, stratify will be fragile — we attempt stratified splits
        stratify_arg = y if y.value_counts().min() >= 2 else None
        if stratify_arg is None:
            warnings.warn(f"Not enough examples to stratify for {name}, performing random split")
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        else:
            # try multiple random states to obtain a test set that contains at least one example
            # of each class when possible. If the minority class is extremely small, fallback to
            # an explicit holdout of one minority sample into the test set.
            df_train = None
            df_test = None
            max_attempts = 10
            for rs in range(max_attempts):
                dtr, dte = train_test_split(df, test_size=0.2, random_state=42 + rs, stratify=stratify_arg)
                if dte['sentiment_label'].nunique() == y.nunique() or y.value_counts().min() * 1.0 / len(df) >= 0.01:
                    df_train, df_test = dtr, dte
                    break
            if df_train is None:
                # fallback: if there's at least one minority sample, explicitly reserve one to test
                minority_label = y.value_counts().idxmin()
                if y.value_counts().get(minority_label, 0) >= 1:
                    one_min = df[df['sentiment_label'] == minority_label].head(1)
                    rest = df.drop(one_min.index)
                    dtr, dte = train_test_split(rest, test_size=0.2, random_state=42)
                    df_train = pd.concat([dtr, one_min]).reset_index(drop=True)
                    df_test = dte.reset_index(drop=True)
                else:
                    # nothing we can do; fall back to a random split
                    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

        # Upsample minority class in training set if heavily imbalanced
        counts = df_train['sentiment_label'].value_counts()
        if len(counts) > 1:
            maj = counts.max()
            minc = counts.min()
            if minc < max(5, maj * 0.2):
                # upsample minority class to match majority
                dfs = []
                for cls, grp in df_train.groupby('sentiment_label'):
                    if len(grp) < maj:
                        grp_up = resample(grp, replace=True, n_samples=maj, random_state=42)
                        dfs.append(grp_up)
                    else:
                        dfs.append(grp)
                df_train_bal = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
                print(f'Upsampled training set for {name}: {counts.to_dict()} -> {df_train_bal["sentiment_label"].value_counts().to_dict()}')
            else:
                df_train_bal = df_train
        else:
            # single-class training set — nothing to balance
            df_train_bal = df_train

        # Vectorize: fit on training text, transform on test
        vect = TfidfVectorizer(max_features=20000)
        Xtr = vect.fit_transform(df_train_bal['tokens_join'].fillna(''))
        ytr = df_train_bal['sentiment_label']
        Xte = vect.transform(df_test['tokens_join'].fillna(''))
        yte = df_test['sentiment_label']

        # MultinomialNB (trained on the balanced training set)
        nb = MultinomialNB()
        nb.fit(Xtr, ytr)
        ypred_nb = nb.predict(Xte)
        rep_nb = classification_report(yte, ypred_nb, digits=4)
        (OUT / f'classification_report_{name}_nb.txt').write_text(rep_nb, encoding='utf-8')

        # LinearSVC with class_weight fallback (also trained on balanced data)
        # If training set is single-class, SVM will fail — catch and record
        svm = LinearSVC(class_weight='balanced')
        try:
            svm.fit(Xtr, ytr)
            ypred_svm = svm.predict(Xte)
            rep_svm = classification_report(yte, ypred_svm, digits=4)
        except Exception as e:
            rep_svm = f'Failed to train/predict SVM for {name}: {e}'
        (OUT / f'classification_report_{name}_svm.txt').write_text(rep_svm, encoding='utf-8')

        print('Wrote classification reports for', name)
