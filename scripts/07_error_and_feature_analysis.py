# coding: utf-8
"""
Error analysis + feature importance extraction
- For each dataset in output/labels_{name}.csv do stratified 5-fold CV
- Collect misclassified samples for SVM and NB, sample up to 30 per dataset per model, save to
  output/error_analysis_{name}_{model}.csv
- Extract top-10 positive / top-10 negative features for LinearSVC (coef_) and top-10 per-class
  features for MultinomialNB (feature_log_prob_)
- Save feature lists to output/feature_importance_{name}_{model}.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import resample
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
FILES = {
    'hotel': OUT / 'labels_hotel.csv',
    'ecommerce': OUT / 'labels_ecommerce.csv',
    'waimai': OUT / 'labels_waimai.csv'
}

N_SPLITS = 5
MAX_SAMPLES = 30
RND = 42
random.seed(RND)

def upsample_train(df_train, label_col='sentiment_label'):
    counts = df_train[label_col].value_counts()
    if len(counts) <= 1:
        return df_train
    maj = counts.max()
    minc = counts.min()
    if minc < max(5, maj * 0.2):
        dfs = []
        for cls, grp in df_train.groupby(label_col):
            if len(grp) < maj:
                grp_up = resample(grp, replace=True, n_samples=maj, random_state=42)
                dfs.append(grp_up)
            else:
                dfs.append(grp)
        dfb = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        return dfb
    return df_train

for name, path in FILES.items():
    if not path.exists():
        print('Missing', path)
        continue
    print('Analyzing', name)
    df = pd.read_csv(path, encoding='utf-8', engine='python')
    if 'tokens_join' not in df.columns or 'sentiment_label' not in df.columns:
        print('File missing tokens_join or sentiment_label:', path)
        continue
    X = df['tokens_join'].fillna('')
    y = df['sentiment_label'].astype(int)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

    mis_samples = {'svm': [], 'nb': []}

    fold = 0
    # We'll fit vectorizer inside each fold to mimic previous pipeline
    for train_idx, test_idx in skf.split(X, y):
        fold += 1
        Xtr_text = X.iloc[train_idx]
        Xte_text = X.iloc[test_idx]
        ytr = y.iloc[train_idx]
        yte = y.iloc[test_idx]

        df_tr = pd.DataFrame({'text': Xtr_text.values, 'sentiment_label': ytr.values})
        df_tr_bal = upsample_train(df_tr, label_col='sentiment_label')
        Xtr_text_bal = df_tr_bal['text']
        ytr_bal = df_tr_bal['sentiment_label']

        vect = TfidfVectorizer(max_features=20000)
        Xtr = vect.fit_transform(Xtr_text_bal)
        Xte = vect.transform(Xte_text)

        # NB
        nb = MultinomialNB()
        try:
            nb.fit(Xtr, ytr_bal)
            ypred_nb = nb.predict(Xte)
        except Exception as e:
            print('NB failed', e)
            ypred_nb = np.zeros(len(yte), dtype=int)

        # SVM
        svm = LinearSVC(class_weight='balanced', max_iter=10000)
        try:
            svm.fit(Xtr, ytr_bal)
            ypred_svm = svm.predict(Xte)
        except Exception as e:
            print('SVM failed', e)
            ypred_svm = np.zeros(len(yte), dtype=int)

        # collect misclassified
        for i, idx in enumerate(test_idx):
            true = int(y.iloc[idx])
            nbp = int(ypred_nb[i])
            svmp = int(ypred_svm[i])
            row = df.iloc[idx]
            base = {'index': int(idx), 'review': row.get('review',''), 'tokens_join': row.get('tokens_join',''), 'true': true, 'fold': fold}
            if nbp != true:
                rec = base.copy(); rec.update({'pred': nbp}); mis_samples['nb'].append(rec)
            if svmp != true:
                rec = base.copy(); rec.update({'pred': svmp}); mis_samples['svm'].append(rec)

    # sample up to MAX_SAMPLES random misclassifications per model
    for model in ['svm','nb']:
        items = mis_samples[model]
        if not items:
            print(f'No misclassifications for {name} {model}')
            continue
        sampled = items if len(items) <= MAX_SAMPLES else random.sample(items, MAX_SAMPLES)
        out_df = pd.DataFrame(sampled)
        out_path = OUT / f'error_analysis_{name}_{model}.csv'
        out_df.to_csv(out_path, index=False, encoding='utf-8')
        print('Wrote', out_path)

    # Feature importance
    # Fit on full data (with upsampling on train split simulated by fitting on whole data with no upsample) to extract features
    vect_full = TfidfVectorizer(max_features=20000)
    X_full = vect_full.fit_transform(X)

    # Fit final models on full data (balance via class_weight for SVM)
    nb_full = MultinomialNB()
    try:
        nb_full.fit(X_full, y)
    except Exception as e:
        print('NB full fit failed', e)
    svm_full = LinearSVC(class_weight='balanced', max_iter=10000)
    try:
        svm_full.fit(X_full, y)
    except Exception as e:
        print('SVM full fit failed', e)

    feature_names = np.array(vect_full.get_feature_names_out())

    # SVM coef: for binary, shape (1, n_features) if labels are [0,1]
    try:
        coefs = svm_full.coef_[0]
        # top positive = supports class 1
        top_pos_idx = np.argsort(coefs)[-10:][::-1]
        top_neg_idx = np.argsort(coefs)[:10]
        svm_feat = pd.DataFrame({
            'pos_word': feature_names[top_pos_idx], 'pos_coef': coefs[top_pos_idx],
            'neg_word': feature_names[top_neg_idx], 'neg_coef': coefs[top_neg_idx]
        })
        svm_feat.to_csv(OUT / f'feature_importance_{name}_svm.csv', index=False, encoding='utf-8')
        print('Wrote', OUT / f'feature_importance_{name}_svm.csv')
    except Exception as e:
        print('SVM feature importance failed', e)

    # NB feature_log_prob_: shape (n_classes, n_features)
    try:
        logprob = nb_full.feature_log_prob_
        # class 1 vs class 0
        class1_idx = np.argsort(logprob[1])[-10:][::-1]
        class0_idx = np.argsort(logprob[0])[-10:][::-1]
        nb_feat = pd.DataFrame({
            'pos_word': feature_names[class1_idx], 'pos_logprob': logprob[1][class1_idx],
            'neg_word': feature_names[class0_idx], 'neg_logprob': logprob[0][class0_idx]
        })
        nb_feat.to_csv(OUT / f'feature_importance_{name}_nb.csv', index=False, encoding='utf-8')
        print('Wrote', OUT / f'feature_importance_{name}_nb.csv')
    except Exception as e:
        print('NB feature importance failed', e)

print('Done error and feature analysis')
