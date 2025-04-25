#!/usr/bin/env python3
import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def main():
    bert = np.load('bert_preds.npy', allow_pickle=True).item()
    gat  = np.load('gat_preds.npy',  allow_pickle=True).item()

    bert_ids = [str(t).strip() for t in bert['tweet_ids']]
    gat_ids  = [str(t).strip() for t in gat ['tweet_ids']]

    bert_idx = {tid:i for i,tid in enumerate(bert_ids)}
    gat_idx  = {tid:i for i,tid in enumerate(gat_ids)}

    shared = sorted(set(bert_idx) & set(gat_idx))
    print(f"Found {len(shared)} tweet_ids in common (BERT: {len(bert_ids)}, GAT: {len(gat_ids)})")
    if not shared:
        return

    bert_p = np.stack([bert['probs'][bert_idx[tid]] for tid in shared])
    gat_p  = np.stack([ gat ['probs'][ gat_idx[tid]] for tid in shared])
    y      = np.array([bert['labels'][bert_idx[tid]] for tid in shared])

    X = np.hstack([bert_p, gat_p])

    # 5‑fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        LogisticRegression(max_iter=1000),
        X, y,
        cv=skf,
        scoring='f1_macro'
    )
    print("=== 5‑Fold CV Macro‑F1 ===")
    for i,sc in enumerate(scores,1):
        print(f" Fold {i}: {sc:.3f}")
    print(f" Mean macro‑F1: {scores.mean():.3f}\n")

    # Train final
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/ensemble_lr.pkl')
    print("Saved final ensemble ➞ models/ensemble_lr.pkl\n")

    # In‑sample report
    preds = clf.predict(X)
    print(classification_report(y, preds))
    print("Confusion matrix:\n", confusion_matrix(y, preds))

if __name__=='__main__':
    main()
