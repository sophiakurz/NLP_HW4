#!/usr/bin/env python3
# plot_tree_uniform.py
# ———— 画成网状、均匀分布的“力导向”图 ————

import os, ast
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# ———— 只需改这三行 ————
RAW_DIR = "/ocean/projects/cis240109p/ywu22/twitter15"
OUT_DIR = "/ocean/projects/cis240109p/ywu22/NLP_Assignment3/fake_news_gnn/visualizations/plots"
# 如果想画最深或指定的源 tweet，填具体 ID，否则留空自动挑最深：
SRC_ID = ""  
# ——————————————————————

def load_best_tree(raw_dir, src_id=""):
    best = (-1, None, None, None)  # (depth, G, root, depth_map)
    for fn in os.listdir(os.path.join(raw_dir, "tree")):
        if not fn.endswith(".txt"): continue
        path = os.path.join(raw_dir, "tree", fn)
        # 建图 + 计算深度
        G = nx.DiGraph()
        root = fn[:-4]
        G.add_node(root)
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if "->" not in ln: continue
                l, r = ln.split("->")
                p, c = ast.literal_eval(l), ast.literal_eval(r)
                u = str(p[1])
                v = str(c[1])
                if u.upper() == 'ROOT' or v.upper() == 'ROOT':
                    continue
                G.add_edge(u, v)
        # BFS深度
        depth = {root: 0}
        q = [root]
        while q:
            u = q.pop(0)
            for v in G.successors(u):
                if v not in depth:
                    depth[v] = depth[u] + 1
                    q.append(v)
        dmax = max(depth.values())
        # 如果指定了 SRC_ID，只取它；否则挑最深
        if (src_id and root == src_id) or (not src_id and dmax > best[0]):
            best = (dmax, G, root, depth)
    if best[1] is None:
        raise RuntimeError("没找到任何符合条件的传播树！")
    return best

def main():
    depth, G, root, depth_map = load_best_tree(RAW_DIR, SRC_ID)
    print(f"Plotting tree {root} (depth={depth}), total nodes={G.number_of_nodes()}")

    # ———— 布局：优先 neato，再 fallback spring ————
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
        print("Layout: graphviz neato")
    except Exception:
        pos = nx.spring_layout(
            G,
            k=1.0 / (G.number_of_nodes()**0.5),
            iterations=200,
            seed=42
        )
        print("Layout: spring_layout")

    # 画图
    plt.figure(figsize=(10, 10))
    # 边
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.3,
        edge_color="gray",
        width=0.7
    )

    # 节点：统一大小，按深度渐变色
    nodelist = list(G.nodes())
    depths = [ depth_map[n] for n in nodelist ]
    cmap = plt.get_cmap("viridis", max(depths) + 1)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist = nodelist,
        node_size = 80,
        node_color = depths,
        cmap = cmap,
        linewidths = 0.3,
        edgecolors = "black"
    )

    # 根节点高亮为红色
    nx.draw_networkx_nodes(
        G, pos,
        nodelist = [root],
        node_size = 200,
        node_color = "red",
        edgecolors = "black",
        linewidths = 1.0
    )

    plt.title(f"Uniform Propagation Graph\nroot={root} (depth={depth})", fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"tree_uniform_{root}.png")
    plt.savefig(out_path, dpi=150)
    print(f"[✓] Saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
