import os, ast, yaml
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# ----------  Read labels / sources ----------

def load_labels(path):
    tid, lbl = [], []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            tag, sid = ln.strip().split(":")
            tid.append(sid)
            lbl.append(tag.lower())
    df = pd.DataFrame({"tweet_id": tid, "label": lbl})
    lbl2id = {l: i for i, l in enumerate(sorted(df.label.unique()))}
    df["y"] = df.label.map(lbl2id)
    return df, lbl2id


def load_sources(path):
    tid, txt = [], []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            i, t = ln.strip().split("\t", 1)
            tid.append(i)
            txt.append(t)
    return pd.DataFrame({"tweet_id": tid, "text": txt})


# ----------  Parse tree files ----------

def parse_tree_file(path):
    """Return list of (parent_uid, child_uid, delta_minutes)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if "->" not in ln:
                continue
            left, right = ln.strip().split("->")
            p_node = ast.literal_eval(left)
            c_node = ast.literal_eval(right)
            dtm = float(c_node[2]) - float(p_node[2])
            out.append((p_node[0], c_node[0], dtm))
    return out


def build_td_bu_edges(tree_dir: str):
    td_edges, bu_edges, dt_list = [], [], []
    edge_cnt = defaultdict(int)   # NEW ⭐ counts occurrences of each (parent_tid, child_tid)

    for file in Path(tree_dir).iterdir():
        with open(file) as f:
            for line in f:
                pa, ch = line.strip().split("->")
                uid_p, tid_p, dt_p = eval(pa)
                uid_c, tid_c, dt_c = eval(ch)

                # --------- [1] Collapse retweet nodes ----------
                if tid_p == tid_c:                # retweet: same tweet ID
                    edge_cnt[(tid_p, tid_c)] += 1  # only increment weight, do not add a new edge
                    continue

                # Otherwise, follow original logic
                td_edges.append((tid_p, tid_c))
                bu_edges.append((tid_c, tid_p))
                dt_list.append(float(dt_c))
                edge_cnt[(tid_p, tid_c)] += 1     # accumulate weight ⭐

    return td_edges, bu_edges, dt_list, edge_cnt


# ----------  Command-line interface ----------

if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw = cfg["raw_dir"]
    work = cfg["work_dir"]
    os.makedirs(work, exist_ok=True)

    # load labels & sources
    df_lab, lbl2id = load_labels(os.path.join(raw, "label.txt"))
    df_src = load_sources(os.path.join(raw, "source_tweets.txt"))

    # parse trees
    td, bu, times, _ = build_td_bu_edges(os.path.join(raw, "tree"))

    # --- Save intermediate files ---
    df_lab.to_csv(os.path.join(work, "labels.csv"), index=False)
    df_src.to_csv(os.path.join(work, "sources.csv"), index=False)
    edge_df = pd.DataFrame({
        "u": [u for u, _ in td],
        "v": [v for _, v in td],
        "dt": times
    })
    edge_df.to_csv(os.path.join(work, "td_edges.csv"), index=False)
    # You can save reverse edges or mapping dicts as needed

    print(f"[✓] Parsed {len(df_lab)} trees and {len(edge_df)} TD edges.")
