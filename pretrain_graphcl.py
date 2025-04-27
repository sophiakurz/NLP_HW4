# pretrain_graphcl.py
# --------------------------------------------------
import os, math, torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

# ---------- 随机增强 ----------------------------------------------------------
def drop_edge_and_attr(ei: torch.Tensor, ea: torch.Tensor, p: float):
    mask = torch.rand(ei.size(1), device=ei.device) > p
    return ei[:, mask], ea[mask]

def mask_x(x: torch.Tensor, p: float):
    mask = torch.rand(x.size(0), device=x.device) < p
    x2 = x.clone();  x2[mask] = 0
    return x2

# ---------- InfoNCE -----------------------------------------------------------
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.05,batch_size=4096):
    """
    整批节点做对比；loss ≈ -log sim⁺ / Σsim  (SimCLR)
    """
    if batch_size is None or batch_size >= z1.size(0):
        idx = torch.arange(z1.size(0), device=z1.device)
    else:
        idx = torch.randperm(z1.size(0), device=z1.device)[:batch_size]
    z1, z2 = z1[idx], z2[idx]  
    N  = z1.size(0)
    z  = torch.cat([z1, z2], dim=0)               # [2N, d]
    sim = torch.mm(z, z.t()) / tau                # 余弦相似度 / τ
    sim.masked_fill_(torch.eye(2*N, device=z.device).bool(), -9e15)

    pos = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, pos)

# ---------- 可视化 ------------------------------------------------------------
def _tsne_plot(z, y, path):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    z_2d = TSNE(perplexity=30, n_iter=1_000).fit_transform(z)
    cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    plt.figure(figsize=(6, 5))
    for lbl in range(-1, y.max().item()+1):
        m = y == lbl
        col = 'lightgray' if lbl == -1 else cmap[lbl]
        plt.scatter(z_2d[m, 0], z_2d[m, 1], s=6, c=col, label=str(lbl), alpha=.6)
    plt.axis('off'); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

# ---------- 一次 epoch --------------------------------------------------------
@torch.no_grad()
def _maybe_viz(epoch, z, y, out_dir, every=10, n_viz=2000):
    if y is None or epoch % every:
        return
    os.makedirs(out_dir, exist_ok=True)
    idx = torch.randperm(z.size(0))[:n_viz]
    _tsne_plot(z[idx].cpu().numpy(), y[idx].cpu(), f"{out_dir}/tsne_Ep{epoch:02d}.png")

def graphcl_epoch(model: torch.nn.Module,
                  data:  HeteroData,
                  opt:   torch.optim.Optimizer,
                  device:        torch.device,
                  *,
                  edge_drop:     float = .3,
                  feat_drop:     float = .1,
                  tau:           float = .05,
                  batch:           int = 2048,
                  y_label: torch.Tensor | None = None,
                  viz_every:     int   = 10,
                  viz_dir:       str   = "tsne"):
    """
    返回当轮 contrastive loss；若给 y_label，则定期做 t-SNE 可视化
    """
    model.train()

    # --- 1) 两份视图 ----------------------------------------------------------
    d1 = data.clone().to(device)
    d2 = data.clone().to(device)

    d1['tweet'].x = mask_x(d1['tweet'].x, feat_drop)
    d2['tweet'].x = mask_x(d2['tweet'].x, feat_drop)

    for et in d1.edge_index_dict.keys():
        d1[et].edge_index, d1[et].edge_attr = drop_edge_and_attr(
            d1[et].edge_index, d1[et].edge_attr, edge_drop)
        d2[et].edge_index, d2[et].edge_attr = drop_edge_and_attr(
            d2[et].edge_index, d2[et].edge_attr, edge_drop)

    # --- 2) 前向  -------------------------------------------------------------
    _, z1, _ = model(d1.x_dict, d1.edge_index_dict, d1.edge_attr_dict)
    _, z2, _ = model(d2.x_dict, d2.edge_index_dict, d2.edge_attr_dict)

    loss = nt_xent(z1, z2, tau)

    # --- 3) 反向  -------------------------------------------------------------
    opt.zero_grad(); loss.backward(); opt.step()

    # --- 4) 可视化 ------------------------------------------------------------
    _maybe_viz(graphcl_epoch.ep, z1, y_label, viz_dir, viz_every)
    graphcl_epoch.ep += 1
    return loss.item()

graphcl_epoch.ep = 1   # 作为静态成员记录 epoch 计数
