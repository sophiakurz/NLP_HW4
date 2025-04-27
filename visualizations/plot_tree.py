#!/usr/bin/env python3
# plot_tree_hier_bfs.py

import os, ast
import networkx as nx
import matplotlib.pyplot as plt

# ———— 修改下面三行 ————
raw_dir = "/ocean/projects/cis240109p/ywu22/twitter15"
out_dir = "/ocean/projects/cis240109p/ywu22/NLP_Assignment3/fake_news_gnn/visualizations/plots"
src_id  = "765934198297362432"   # 你想画的那条 source tweet ID
# ——————————————————————

tree_file = os.path.join(raw_dir, "tree", f"{src_id}.txt")
if not os.path.exists(tree_file):
    raise FileNotFoundError(f"找不到 {tree_file}")

# 1. 读文件并构造有向图，只保留真正的 tweet_id
G = nx.DiGraph()
with open(tree_file, encoding="utf-8") as f:
    for ln in f:
        if "->" not in ln: continue
        left, right = ln.strip().split("->")
        p = ast.literal_eval(left);   c = ast.literal_eval(right)
        pid, cid = str(p[1]), str(c[1])
        if pid == "ROOT" or cid == "ROOT":
            continue
        G.add_edge(pid, cid)

# 2. BFS 计算每个节点到 root 的距离（depth）
depth = nx.single_source_shortest_path_length(G, src_id)

# 把同一深度的节点分层
layers = {}
for node, d in depth.items():
    layers.setdefault(d, []).append(node)

# 3. 给每层节点按横轴等距分布，纵坐标 = -depth * vert_gap
pos = {}
width = 1.0
vert_gap = 1.0
for d, nodes in layers.items():
    nodes = sorted(nodes)
    n = len(nodes)
    for i, node in enumerate(nodes):
        x = (i + 0.5) / n * width   # [0,1] 上等距
        y = -d * vert_gap
        pos[node] = (x, y)

# 4. 绘图
plt.figure(figsize=(12, 8))
others = [n for n in G.nodes if n != src_id]
nx.draw_networkx_nodes(
    G, pos,
    nodelist=others,
    node_size=150,
    node_color="white",
    edgecolors="black",
    linewidths=0.5,
)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=[src_id],
    node_size=400,
    node_color="red",
    label="source tweet",
)
nx.draw_networkx_edges(
    G, pos,
    arrowstyle="->",
    arrowsize=8,
    edge_color="gray",
    alpha=0.7,
    connectionstyle="arc3,rad=0.1",
)
plt.title(f"Hierarchical Propagation Tree\nroot={src_id} (depth={max(depth.values())})",
          fontsize=14)
plt.axis("off")

# 5. 保存 & 展示
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"tree_hier_{src_id}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"[✓] Saved figure to {out_path}")
plt.show()
