#!/usr/bin/env python3
import os
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch_geometric.data import Data
from fake_news_gnn import build_graph, FakeNewsGAT, clean_text

def load_bert_model(model_dir, device):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    mdl.eval()
    return tok, mdl

def bert_probs(texts, tokenizer, model, device, batch_size=16):
    all_logits = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i+batch_size],
            padding=True, truncation=True, max_length=128,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            all_logits.append(model(**enc).logits.cpu())
    logits = torch.cat(all_logits, dim=0)
    return torch.softmax(logits, dim=1).numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tweets',     required=True,
                   help="CSV with columns tweet_id,text")
    p.add_argument('--bert_dir',   default='models/bert_final')
    p.add_argument('--gat_ckpt',   default='models/gat_final.pt')
    p.add_argument('--ensemble',   default='models/ensemble_lr.pkl')
    p.add_argument('--le_encoder', required=True,
                   help="models/le.pkl")
    p.add_argument('--tfidf',      required=True,
                   help="models/tfidf.pkl")
    p.add_argument('--min_df',     type=int,   default=5)
    p.add_argument('--max_df',     type=float, default=0.8)
    p.add_argument('--window_size',type=int,   default=5)
    p.add_argument('--hidden',     type=int,   default=128)
    p.add_argument('--heads',      type=int,   default=4)
    p.add_argument('--dropout',    type=float, default=0.6)
    args = p.parse_args()

    # load everything
    le  = joblib.load(args.le_encoder)
    vec = joblib.load(args.tfidf)
    clf = joblib.load(args.ensemble)

    # read new data & clean
    df = pd.read_csv(args.tweets, dtype=str)
    df['clean_text'] = df['text'].apply(clean_text)
    texts = df['text'].tolist()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) BERT probs
    tok, bmodel = load_bert_model(args.bert_dir, device)
    bert_p = bert_probs(texts, tok, bmodel, device)

    # 2) Build graph & GAT probs
    #    build_graph returns (X, edge_index, edge_weight, vectorizer)
    X, edge_index, edge_weight, _ = build_graph(
        df,
        vectorizer=vec,
        min_df=args.min_df,
        max_df=args.max_df,
        window_size=args.window_size
    )
    gat = FakeNewsGAT(
        in_channels  = X.shape[1],
        hidden_channels = args.hidden,
        out_channels    = len(le.classes_),
        heads        = args.heads,
        dropout      = args.dropout
    ).to(device)
    gat.load_state_dict(torch.load(args.gat_ckpt, map_location=device))
    gat.eval()

    data = Data(
        x         = torch.tensor(X, dtype=torch.float32).to(device),
        edge_index= torch.tensor(edge_index, dtype=torch.long).to(device),
        edge_attr = torch.tensor(edge_weight, dtype=torch.float32).to(device)
    )
    with torch.no_grad():
        logits = gat(data.x, data.edge_index, data.edge_attr)
    gat_p = torch.softmax(logits[:len(df)], dim=1).cpu().numpy()

    # 3) Ensemble
    X_ens = np.hstack([bert_p, gat_p])
    preds = clf.predict(X_ens)
    df['pred_label'] = le.inverse_transform(preds)

    # 4) Save
    out_csv = 'new_predictions.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

if __name__ == '__main__':
    main()
