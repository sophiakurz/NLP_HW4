# build_hetero.py
# --------------------------------------------------
# Build a heterogeneous graph for Twitter-15/16
# edge_attr layout = [w_norm, Δt_norm, infl_flag, cos_sim] (4 dims)
# --------------------------------------------------
import os, argparse, yaml, math
import torch, numpy as np, pandas as pd
from torch_geometric.data import HeteroData
from data_utils import load_labels, load_sources, build_td_bu_edges
from features    import TextEncoder
from collections import Counter
import networkx as nx
from torch.nn.functional import normalize

# --------------------------------------------------
def build_hetero(raw_dir: str,
                 save_path: str,
                 *,
                 device: str = "cpu",
                 max_k: int = 0,
                 keep_bu: bool = True) -> None:

    # ---------- 1. load labels & source tweets -------------------------------
    print("[1/7] Loading labels and sources…")
    lbl_df, lbl2id = load_labels(f"{raw_dir}/label.txt")
    src_df         = load_sources(f"{raw_dir}/source_tweets.txt")
    src_df         = src_df[src_df.tweet_id.isin(lbl_df.tweet_id)].reset_index(drop=True)
    print(f"    Loaded {len(lbl_df)} labels, {len(src_df)} source texts")

    # ---------- 2. parse tree files (TD/BU edges after collapsing retweets) --
    print("[2/7] Building raw edges from tree files…")
    td_raw, bu_raw, dt_raw, edge_cnt = build_td_bu_edges(f"{raw_dir}/tree")
    print(f"    Extracted {len(td_raw)} TD edges, {len(bu_raw)} BU edges")

    # ---------- 3. build node id map ----------------------------------------
    print("[3/7] Collecting tweet IDs & init features…")
    all_tid = set(src_df.tweet_id.astype(str))
    for u, v in td_raw:
        all_tid.update([str(u), str(v)])
    N = len(all_tid)
    tid2nid = {tid: i for i, tid in enumerate(sorted(all_tid))}
    print(f"    Total nodes = {N}")
    x_tweet = torch.zeros((N, 769), dtype=torch.float32)

    # ---------- 4. encode source-tweet text ----------------------------------
    if len(src_df):
        print(f"[4/7] Encoding source texts ({len(src_df)})…")
        encoder  = TextEncoder(device=device)
        src_nids = [tid2nid[str(t)] for t in src_df.tweet_id.astype(str)]
        x_src    = encoder.encode_batch(src_df.text.tolist()).float()
        x_tweet[src_nids] = x_src

    # ===== 4.1 propagate embeddings from parents to children (retweets) =====
    print("    Propagating embeddings to retweets …")
    thr = 1e-6
    have_emb = (x_tweet.abs().sum(dim=1) > thr)            # [N] bool

    # parent → children dict (TD only)
    parent2child = {}
    for (p_tid, c_tid), _ in zip(td_raw, dt_raw):
        pn, cn = tid2nid[str(p_tid)], tid2nid[str(c_tid)]
        parent2child.setdefault(pn, []).append(cn)

    # BFS for at most K hops
    K = 5
    for _ in range(K):
        new = []
        for p, childs in parent2child.items():
            if not have_emb[p]:
                continue
            for c in childs:
                if not have_emb[c]:
                    x_tweet[c] = x_tweet[p]   # copy vector
                    new.append(c)
        if not new:
            break
        have_emb[new] = True

    missing = (~have_emb).sum().item()
    print(f"      after propagation UNK left = {missing}")

    # remaining isolated nodes → random UNK
    if missing:
        torch.manual_seed(0)
        unk_vec = torch.randn(769) * 0.02
        x_tweet[~have_emb] = unk_vec

    # ---------- 5. construct TD/BU edges & basic edge attributes ------------
    print(f"[5/7] Constructing edges (max_k={max_k}, keep_bu={keep_bu})…")
    bucket, td, bu, dt_keep, w_keep = {}, [], [], [], []
    for (u, v), dt in zip(td_raw, dt_raw):
        bucket.setdefault(str(u), []).append((str(v), float(dt)))
    for u, lst in bucket.items():
        if max_k > 0:
            lst = sorted(lst, key=lambda x: x[1])[:max_k]
        for v, dt in lst:
            td.append([tid2nid[u], tid2nid[v]])
            bu.append([tid2nid[v], tid2nid[u]])
            dt_keep.append(abs(dt))
            w_keep.append(edge_cnt[(u, v)])

    td_e = torch.tensor(td, dtype=torch.long).t().contiguous()
    bu_e = torch.tensor(bu, dtype=torch.long).t().contiguous() if keep_bu else None

    # edge_weight / Δt → log-scale normalisation to (0,1]
    edge_w_raw  = torch.tensor(w_keep, dtype=torch.float32)
    edge_w_norm = torch.log1p(edge_w_raw) / torch.log1p(edge_w_raw.max() + 1)
    edge_w_norm = edge_w_norm.unsqueeze(1)

    edge_dt_raw  = torch.tensor(dt_keep, dtype=torch.float32)
    edge_dt_norm = torch.log1p(edge_dt_raw) / math.log1p(60*24*7)   # 7 days cap
    edge_dt_norm = edge_dt_norm.unsqueeze(1)

    print(f"    Final TD edges: {td_e.size(1)}, BU edges: {bu_e.size(1) if bu_e is not None else 0}")

    # ---------- 6. labels & train mask --------------------------------------
    print("[6/7] Setting labels & mask…")
    y = torch.full((N,), -1, dtype=torch.long)
    train_mask = torch.zeros(N, dtype=torch.bool)
    for _, row in lbl_df.iterrows():
        nid = tid2nid[str(row.tweet_id)]
        y[nid] = lbl2id[row.label]
        train_mask[nid] = True
    print(f"    Labeled nodes = {train_mask.sum().item()}")

    # ---------- 7. assemble HeteroData + extra features ---------------------
    print("[7/7] Assembling HeteroData & injecting extra features…")
    data = HeteroData()
    data['tweet'].x          = x_tweet
    data['tweet'].y          = y
    data['tweet'].train_mask = train_mask
    data['tweet','TD','tweet'].edge_index = td_e
    if keep_bu:
        data['tweet','BU','tweet'].edge_index = bu_e

    # 7.1 structural node feats: degree + PageRank
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(N))
    G_nx.add_edges_from(list(zip(td_e[0].tolist(), td_e[1].tolist())))
    deg = np.array([G_nx.degree[i] for i in range(N)], dtype=np.float32)[:, None]
    pr_dict = nx.pagerank(G_nx, alpha=0.85, tol=1e-4)
    pr_raw  = np.array([pr_dict[i] for i in range(N)], dtype=np.float32)
    pr_norm = (pr_raw - pr_raw.min()) / (pr_raw.max() - pr_raw.min() + 1e-9)
    pr_norm = pr_norm[:, None]
    data['tweet'].x = torch.cat([data['tweet'].x,
                                 torch.from_numpy(deg),
                                 torch.from_numpy(pr_norm)], dim=1)

    # 7.2 extra edge_attr: infl_flag & cos_sim
    emb = normalize(data['tweet'].x, dim=1)
    srcs, dsts = td_e
    cos_sim = (emb[srcs] * emb[dsts]).sum(dim=1, keepdim=True)

    pr_torch = torch.from_numpy(pr_norm.squeeze())
    infl_flag = (pr_torch[srcs] > pr_torch[dsts]).float().unsqueeze(1)

    edge_attr_td = torch.cat([edge_w_norm, edge_dt_norm, infl_flag, cos_sim], dim=1)
    data['tweet','TD','tweet'].edge_attr = edge_attr_td

    if keep_bu:
        srcs_b, dsts_b = bu_e
        cos_sim_b  = (emb[srcs_b] * emb[dsts_b]).sum(dim=1, keepdim=True)
        infl_flag_b = (pr_torch[srcs_b] > pr_torch[dsts_b]).float().unsqueeze(1)
        edge_attr_bu = torch.cat([edge_w_norm, edge_dt_norm, infl_flag_b, cos_sim_b], dim=1)
        data['tweet','BU','tweet'].edge_attr = edge_attr_bu

    # 7.3 graph-level stats
    data.num_edges  = torch.tensor([td_e.size(1)], dtype=torch.long)
    data.avg_degree = torch.tensor([2.0 * td_e.size(1) / N], dtype=torch.float32)
    cnts = Counter(lbl_df.label.values)
    dist = torch.tensor([cnts[l] for l,_ in sorted(lbl2id.items(), key=lambda x:x[1])],
                        dtype=torch.float32)
    data.label_dist = dist / dist.sum()

    print(f"    x.shape={data['tweet'].x.shape}, edge_attr.shape={data['tweet','TD','tweet'].edge_attr.shape}")

    # ---------- 8. save ------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'data': data, 'lbl2id': lbl2id}, save_path)
    print(f"[✓] Hetero graph saved to {save_path}")

# --------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg  = yaml.safe_load(open(args.config))
    build_hetero(raw_dir   = cfg['raw_dir'],
                 save_path = os.path.join(cfg['work_dir'], 'hetero_graph.pt'),
                 device    = cfg.get('device','cpu'),
                 max_k     = cfg.get('edge_k', 0),
                 keep_bu   = cfg.get('keep_bu', True))
