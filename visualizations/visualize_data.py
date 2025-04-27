#!/usr/bin/env python3
# visualize_data.py

import os
import ast
import collections
import matplotlib.pyplot as plt

# --------- 在此处直接指定数据目录和输出目录 ---------
raw_dir = "/ocean/projects/cis240109p/ywu22/twitter15"
out_dir = "/ocean/projects/cis240109p/ywu22/NLP_Assignment3/fake_news_gnn/visualizations/plots"
# --------------------------------------------------

# 去掉 argparse，直接运行脚本即可

def load_label_map(path):
    lm = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            tag, tid = ln.strip().split(":")
            lm[tid] = tag.lower()
    return lm


def load_source_ids(path):
    ids = set()
    with open(path, encoding="utf-8") as f:
        for ln in f:
            tid, _ = ln.strip().split("\t", 1)
            ids.add(tid)
    return ids


def parse_trees(tree_dir):
    """返回: 
       deg_counts (list of root 出度), 
       time_delays (list of all Δt),
       sizes (每棵树节点数), depths (每棵树最大深度)"""
    roots = []                    # ← 新增
    deg_counts, time_delays, sizes, depths = [], [], [], []

    for fn in os.listdir(tree_dir):
        if not fn.endswith(".txt"): continue
        root = os.path.splitext(fn)[0]
        roots.append(root) 
        path = os.path.join(tree_dir, fn)

        adj = collections.defaultdict(list)
        nodes = set([root])
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if "->" not in ln: continue
                left, right = ln.strip().split("->")
                p = ast.literal_eval(left); c = ast.literal_eval(right)
                u, v = str(p[1]), str(c[1])
                dt = float(c[2]) - float(p[2])
                time_delays.append(dt)
                adj[u].append(v)
                nodes.update([u, v])

        # ① 出度 = root 直连孩子数
        deg_counts.append(len(adj.get(root, [])))
        # ② 子树规模
        sizes.append(len(nodes))
        # ③ 最大深度（BFS）
        max_d, q, seen = 0, [(root, 0)], {root}
        while q:
            u, d = q.pop(0)
            max_d = max(max_d, d)
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    q.append((v, d+1))
        depths.append(max_d)

    return roots, deg_counts, time_delays, sizes, depths 


def plot_and_save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def main():
    # 修正文件名为实际的 label.txt 和 source_tweets.txt
    label_txt  = os.path.join(raw_dir, "label.txt")
    source_txt = os.path.join(raw_dir, "source_tweets.txt")
    tree_dir   = os.path.join(raw_dir, "tree")

    # 1. ID 对齐检查
    labels = load_label_map(label_txt)
    sources= load_source_ids(source_txt)
    trees  = {os.path.splitext(fn)[0] for fn in os.listdir(tree_dir) if fn.endswith(".txt")}
    print("In label not in source:", set(labels) - sources)
    print("In source not in label:", sources - set(labels))
    print("Trees not in label:", trees - set(labels))
    print("Trees not in source:", trees - sources)

    # 2. 解析 tree 结构
    roots, degs, dts, sizes, depths = parse_trees(tree_dir)
    # 找出 depth 最大的前 5 棵
    top5 = sorted(zip(roots, depths), key=lambda x: -x[1])[:5]
    print("Top 5 deepest trees (tweet_id, depth):")
    for tid, dep in top5:
        print(f"  {tid}: depth={dep}")

    # 3. 画图并保存
    cnt = collections.Counter(labels.values())
    fig = plt.figure(); plt.bar(cnt.keys(), cnt.values()); plt.title("Label Distribution"); plt.xticks(rotation=45)
    plot_and_save(fig, out_dir, "label_distribution")

    fig = plt.figure(); plt.hist(degs, bins=30); plt.title("Root Out‑degree Distribution")
    plot_and_save(fig, out_dir, "out_degree")

    dt_good = [d for d in dts if d < 60]
    fig = plt.figure()
    plt.hist(dt_good, bins=30)
    plt.title("Edge Δt Distribution (< 60 min)")
    plot_and_save(fig, out_dir, "time_delay_lt60")

    fig = plt.figure(); plt.hist(sizes, bins=30); plt.title("Subtree Size Distribution")
    plot_and_save(fig, out_dir, "subtree_size")

    fig = plt.figure(); plt.hist(depths, bins=30); plt.title("Max Depth Distribution")
    plot_and_save(fig, out_dir, "max_depth")

    fig = plt.figure(); plt.scatter(sizes, depths, alpha=0.6); plt.title("Size vs Depth"); plt.xlabel("Size"); plt.ylabel("Depth")
    plot_and_save(fig, out_dir, "size_vs_depth")

if __name__ == '__main__':
    main()
