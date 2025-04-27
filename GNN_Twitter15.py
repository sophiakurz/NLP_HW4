#!/usr/bin/env python3
"""
gnn_training_accuracy_plot.py

Train multiple GNN architectures on Twitter15 data, log per-epoch train/val accuracy
to CSV files (one per model), select the best model by test accuracy, and plot its
accuracy curves.
"""
import os
import random
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ARMAConv, SGConv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import nltk
from nltk.corpus import wordnet

# Download WordNet data (for synonym replacement)
nltk.download('wordnet', quiet=True)

# -----------------------------------------------------------------------------
# Utility functions for text cleaning & graph construction
# -----------------------------------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def load_data(source_file, label_file):
    tweet_ids, tweets = [], []
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            tweet_ids.append(parts[0])
            tweets.append(clean_text(parts[1]))
    labels = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            label, tid = line.split(':', 1)
            labels[tid] = label.lower()
    filtered_ids, filtered_texts, filtered_labels = [], [], []
    for tid, txt in zip(tweet_ids, tweets):
        if tid in labels:
            filtered_ids.append(tid)
            filtered_texts.append(txt)
            filtered_labels.append(labels[tid])
    return filtered_ids, filtered_texts, filtered_labels

def get_synonyms(word):
    syns = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_',' ')
            if name != word and name not in syns:
                syns.append(name)
    return syns

def synonym_replacement(text, n=1):
    words = text.split()
    if len(words) <= 1: return text
    candidates = [w for w in words if get_synonyms(w)]
    if not candidates: return text
    to_replace = random.sample(candidates, min(n, len(candidates)))
    new_words = [random.choice(get_synonyms(w)) if w in to_replace else w for w in words]
    return ' '.join(new_words)

def perform_data_augmentation(texts, labels):
    counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    max_count = max(counts.values())
    aug_texts, aug_labels = [], []
    for lbl, cnt in counts.items():
        if cnt >= max_count * 0.9: continue
        indices = [i for i,l in enumerate(labels) if l==lbl]
        num_aug = min(int(max_count*0.9)-cnt, len(indices)*2)
        for _ in range(num_aug):
            idx = random.choice(indices)
            aug_texts.append(synonym_replacement(texts[idx], n=random.randint(1,3)))
            aug_labels.append(lbl)
    return texts + aug_texts, labels + aug_labels

def create_graph_features(texts, k=5):
    # TF-IDF features for text nodes
    vec = TfidfVectorizer(max_features=2500, ngram_range=(1,3), min_df=2, max_df=0.9, sublinear_tf=True)
    X_dt = vec.fit_transform(texts).toarray()
    num_docs, num_terms = X_dt.shape

    # Build bipartite doc-term edges
    rows, cols, vals = [], [], []
    for d in range(num_docs):
        for t in np.nonzero(X_dt[d])[0]:
            w = X_dt[d,t]
            rows += [d, num_docs+t]
            cols += [num_docs+t, d]
            vals += [w, w]
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weight = torch.tensor(vals, dtype=torch.float)

    # Node features: TF-IDF rows + identity for term nodes
    feats = torch.tensor(np.vstack([X_dt, np.eye(num_terms)]), dtype=torch.float)
    return feats, edge_index, edge_weight, vec

# -----------------------------------------------------------------------------
# GNN architecture definitions
# -----------------------------------------------------------------------------

class BiGCN_A(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_c, hid_c)
        self.conv2 = GCNConv(hid_c, out_c)
        self.dropout = dropout
    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

class BiGAT(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, heads=8, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_c, hid_c, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hid_c*heads, out_c, heads=1, concat=False, dropout=dropout)
    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class BiSAGE(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_c, hid_c)
        self.conv2 = SAGEConv(hid_c, out_c)
        self.dropout = dropout
    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

class BiARMA(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, stacks=1, layers=1, dropout=0.5):
        super().__init__()
        self.conv1 = ARMAConv(in_c, hid_c, num_stacks=stacks, num_layers=layers, shared_weights=True, dropout=dropout)
        self.conv2 = ARMAConv(hid_c, out_c, num_stacks=stacks, num_layers=layers, shared_weights=True, dropout=dropout)
    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.conv1.dropout, training=self.training)
        return self.conv2(x, edge_index)

class BiSGCN(torch.nn.Module):
    def __init__(self, in_c, out_c, K=2):
        super().__init__()
        self.conv = SGConv(in_c, out_c, K=K)
    def forward(self, x, edge_index, edge_attr=None):
        return self.conv(x, edge_index)

# -----------------------------------------------------------------------------
# Training, testing, and plotting
# -----------------------------------------------------------------------------

def train_model(model, data, labels_np, train_mask, val_mask,
                epochs=200, patience=20, metrics_csv="metrics.csv"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = model.to(device), data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    sched = CosineAnnealingWarmRestarts(opt, T_0=50)

    # CSV header
    with open(metrics_csv, "w") as f:
        f.write("epoch,train_accuracy,val_accuracy\n")

    best_loss, wait, best_state = float('inf'), 0, None

    for ep in range(1, epochs+1):
        # train
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index, getattr(data, 'edge_attr', None))
        loss = F.cross_entropy(out[train_mask],
                               torch.tensor(labels_np[train_mask.cpu().numpy()], device=device))
        loss.backward(); opt.step(); sched.step()

        # eval
        model.eval()
        with torch.no_grad():
            preds = out.argmax(dim=1)
            train_acc = (preds[train_mask] ==
                         torch.tensor(labels_np[train_mask.cpu().numpy()], device=device)
                        ).float().mean().item()

            val_out = model(data.x, data.edge_index, getattr(data, 'edge_attr', None))
            val_preds = val_out[val_mask].argmax(dim=1)
            val_acc = (val_preds.cpu().numpy() ==
                       labels_np[val_mask.cpu().numpy()]
                      ).mean()

        # log metrics
        with open(metrics_csv, "a") as f:
            f.write(f"{ep},{train_acc:.4f},{val_acc:.4f}\n")

        # early stopping
        if loss.item() < best_loss:
            best_loss, best_state, wait = loss.item(), model.state_dict(), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model

def test_model(model, data, test_mask, true_labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = model.to(device), data.to(device)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, getattr(data, 'edge_attr', None))
        preds = out[test_mask].argmax(dim=1)
    return (preds.cpu().numpy() == true_labels).mean()

def main():
    # fix seeds
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    # load & preprocess
    src = 'twitter_data/twitter15/twitter15_source_tweets.txt'
    lbl = 'twitter_data/twitter15/twitter15_label.txt'
    ids, texts, labels = load_data(src, lbl)
    texts, labels = perform_data_augmentation(texts, labels)
    feats, e_idx, e_wt, _ = create_graph_features(texts)
    data = Data(x=feats, edge_index=e_idx, edge_attr=e_wt)

    # masks
    labels_np = np.array([{'false':0,'non-rumor':1,'true':2,'unverified':3}[l] for l in labels])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, tmp_idx = next(sss.split(np.zeros(len(labels_np)), labels_np))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(sss2.split(tmp_idx, labels_np[tmp_idx]))
    train_mask = torch.tensor(tr_idx, dtype=torch.long)
    val_mask   = torch.tensor(tmp_idx[val_idx], dtype=torch.long)
    test_mask  = torch.tensor(tmp_idx[test_idx], dtype=torch.long)

    # architectures to try
    in_c = feats.shape[1]
    hidden = 128
    num_classes = len(set(labels_np))
    archs = {
        "BiGCN_A": lambda: BiGCN_A(in_c, hidden, num_classes),
        "BiGAT":   lambda: BiGAT(in_c, hidden, num_classes, heads=8),
        "BiSAGE":  lambda: BiSAGE(in_c, hidden, num_classes),
        "BiARMA":  lambda: BiARMA(in_c, hidden, num_classes, stacks=1, layers=1),
        "BiSGCN":  lambda: BiSGCN(in_c, num_classes, K=2),
    }

    # train & evaluate all
    test_accuracies = {}
    for name, make_model in archs.items():
        print(f"\n=== Training {name} ===")
        csv_file = f"metrics_{name}.csv"
        model = train_model(make_model(), data, labels_np,
                            train_mask, val_mask,
                            epochs=200, patience=20,
                            metrics_csv=csv_file)
        acc = test_model(model, data, test_mask, labels_np[test_mask.cpu().numpy()])
        test_accuracies[name] = acc
        print(f"{name} test acc: {acc*100:.2f}%")

    # pick best
    best = max(test_accuracies, key=test_accuracies.get)
    print(f"\nBest model: {best} ({test_accuracies[best]*100:.2f}% test acc)")

    # plot its CSV
    df = pd.read_csv(f"metrics_{best}.csv")
    plt.figure()
    plt.plot(df["epoch"], df["train_accuracy"], marker="o", label="Train Acc")
    plt.plot(df["epoch"], df["val_accuracy"],   marker="o", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Replication: Accuracy Curves for {best}")
    plt.legend()
    plt.tight_layout()
    out_png = f"accuracy_{best}.png"
    plt.savefig(out_png)
    plt.show()
    print(f"Saved plot ➞ {out_png}")

if __name__ == "__main__":
    main()
