#!/usr/bin/env python3
import os
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class FakeNewsGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads,
                             out_channels,
                             heads=1,
                             concat=False,
                             dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x

def clean_text(s: str) -> str:
    return s.lower().strip()

def load_dataset(src_path, lbl_path):
    df_txt = pd.read_csv(src_path, sep='\t',
                         names=['tweet_id','text'], dtype=str)
    df_txt['clean_text'] = df_txt['text'].apply(clean_text)
    df_lbl = pd.read_csv(lbl_path, sep=':', names=['label','tweet_id'], dtype=str)
    df = df_txt.merge(df_lbl, on='tweet_id')
    le = LabelEncoder()
    df['y_doc'] = le.fit_transform(df['label'])
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/le.pkl')
    print("Saved label encoder ➞ models/le.pkl")
    return df

def build_graph(df, vectorizer=None, min_df=5, max_df=0.8, window_size=5):
    texts = df['clean_text'].tolist()
    if vectorizer is None:
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
        X_dt = vectorizer.fit_transform(texts).toarray()
    else:
        X_dt = vectorizer.transform(texts).toarray()

    num_docs, num_terms = X_dt.shape
    rows, cols, vals = [], [], []
    for d in range(num_docs):
        for t in np.nonzero(X_dt[d])[0]:
            w = X_dt[d, t]
            # doc → term
            rows.append(d);           cols.append(num_docs + t); vals.append(w)
            # term → doc
            rows.append(num_docs + t); cols.append(d);             vals.append(w)

    edge_index  = np.vstack([rows, cols])
    edge_weight = np.array(vals, dtype=np.float32)
    # features: docs get TF–IDF, terms get one‑hot
    X = np.vstack([X_dt, np.eye(num_terms)])
    return X, edge_index, edge_weight, vectorizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src',    required=True,
                   help="tweet_id<TAB>text")
    p.add_argument('--labels', required=True,
                   help="label:tweet_id")
    p.add_argument('--min_df',   type=int,   default=5)
    p.add_argument('--max_df',   type=float, default=0.8)
    p.add_argument('--window',   type=int,   default=5)
    p.add_argument('--hidden',   type=int,   default=128)
    p.add_argument('--heads',    type=int,   default=4)
    p.add_argument('--dropout',  type=float, default=0.6)
    p.add_argument('--lr',       type=float, default=0.005)
    p.add_argument('--epochs',   type=int,   default=200)
    args = p.parse_args()

    # 1) Load & label‐encode
    df = load_dataset(args.src, args.labels)

    # 2) Build doc–term graph and save TF–IDF
    X, edge_index, edge_weight, vec = build_graph(
        df,
        vectorizer=None,
        min_df=args.min_df,
        max_df=args.max_df,
        window_size=args.window
    )
    joblib.dump(vec, 'models/tfidf.pkl')
    print("Saved TF–IDF vectorizer ➞ models/tfidf.pkl")

    # 3) Create PyG Data (only x, edge_index, edge_attr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Data(
        x=torch.tensor(X, dtype=torch.float32).to(device),
        edge_index=torch.tensor(edge_index, dtype=torch.long).to(device),
        edge_attr=torch.tensor(edge_weight, dtype=torch.float32).to(device),
    )

    # 4) Split **only** the document nodes
    num_docs = len(df)
    idx = np.arange(num_docs)
    np.random.shuffle(idx)
    train_cut = int(0.8 * num_docs)
    val_cut   = int(0.9 * num_docs)
    doc_train_idx = idx[:train_cut]
    doc_val_idx   = idx[train_cut:val_cut]
    doc_test_idx  = idx[val_cut:]

    # 5) Prepare label tensor for docs
    y = torch.tensor(df['y_doc'].values, dtype=torch.long).to(device)

    # 6) Instantiate GAT
    model = FakeNewsGAT(
        in_channels=X.shape[1],
        hidden_channels=args.hidden,
        out_channels=y.max().item()+1,
        heads=args.heads,
        dropout=args.dropout
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 7) Training loop
    with open("metrics.csv","w") as f:
        f.write("epoch,train_accuracy,val_accuracy\n")
    for epoch in range(1, args.epochs+1):
        model.train()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = F.cross_entropy(out[doc_train_idx], y[doc_train_idx])
        loss.backward()
        opt.step()
        opt.zero_grad()
        model.eval()
        with torch.no_grad():
            pred      = out.argmax(dim=1)
            train_acc = (pred[doc_train_idx] == y[doc_train_idx]).float().mean().item()
            val_acc   = (pred[doc_val_idx]   == y[doc_val_idx]).float().mean().item()
        with open("metrics.csv","a") as f:
            f.write(f"{epoch},{train_acc:.4f},{val_acc:.4f}\n")
        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # 8) Save
    torch.save(model.state_dict(), 'models/gat_final.pt')
    print("Saved GAT model ➞ models/gat_final.pt")

    # 9) Final test‐set report
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        pred = out.argmax(dim=1)[doc_test_idx].cpu().numpy()
        true = y[doc_test_idx].cpu().numpy()
    from sklearn.metrics import classification_report
    print(classification_report(true, pred, digits=4))

if __name__=='__main__':
    main()