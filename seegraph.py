import torch
import pandas as pd

# 1) 载入保存好的 hetero_graph.pt  ----------
pt_path = "/ocean/projects/cis240109p/ywu22/NLP_Assignment3/fake_news_gnn/work/twitter15/hetero_graph.pt"
bundle  = torch.load(pt_path,weights_only = False)
data    = bundle["data"]                    # torch_geometric.data.HeteroData

# 2) 看整体结构 ----------------------------
print(data)                                 # 节点类型、边类型、特征维度、边数

# 3) 节点：抽 5 个看看前 10 维特征 ----------
node_ids = list(range(5))
x_sample = data["tweet"].x[node_ids, :10]   # [5, 10]
print("\nnode feature sample (first 10 dims):")
print(pd.DataFrame(x_sample.numpy(), index=node_ids))

# 4) 边：先看 TD 关系的前 5 条 --------------
edge_index = data["tweet", "TD", "tweet"].edge_index
edge_attr  = data["tweet", "TD", "tweet"].edge_attr      # [E, 2] | Δt + cos_sim

rows = []
for i in range(5):
    src, dst = edge_index[:, i].tolist()
    dt, cos  = edge_attr[i].tolist()
    rows.append({"eid": i, "src": src, "dst": dst,
                 "Δt(min)": round(dt, 2), "cos_sim": round(cos, 3)})
print("\nfirst 5 TD edges:")
print(pd.DataFrame(rows))

print(data['tweet'].x[node_ids, -2:])   # degree & pagerank

# 随机抽 5 个有文本的节点
has_text = (data['tweet'].x[:, :769].abs().sum(dim=1) > 0)
idx = torch.nonzero(has_text)[:5].squeeze()
print(data['tweet'].x[idx, :10])      
