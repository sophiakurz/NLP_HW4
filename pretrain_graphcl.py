# pretrain_graphcl.py
# --------------------------------------------------
import os, math, torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

# ---------- Random augmentations ------------------------------------------------

def drop_edge_and_attr(ei: torch.Tensor, ea: torch.Tensor, p: float):
    """Randomly drop edges together with their attributes with probability *p*."""
    mask = torch.rand(ei.size(1), device=ei.device) > p
    return ei[:, mask], ea[mask]


def mask_x(x: torch.Tensor, p: float):
    """Randomly mask node features with probability *p*."""
    mask = torch.rand(x.size(0), device=x.device) < p
    x2 = x.clone();  x2[mask] = 0
    return x2

# ---------- InfoNCE ----------------------------------------------------------------

def nt_xent(
    z1: torch.Tensor,
    z2: torch.Tensor,
    tau: float = 0.05,
    batch_size: int | None = 4096,
):
    """Contrastive loss over the entire batch (SimCLR style).

    The loss is approximately ``-log(sim⁺ / Σ sim)`` where ``sim`` is the cosine
    similarity. A smaller temperature *tau* sharpens the distribution.
    """
    if batch_size is None or batch_size >= z1.size(0):
        idx = torch.arange(z1.size(0), device=z1.device)
    else:
        idx = torch.randperm(z1.size(0), device=z1.device)[:batch_size]
    z1, z2 = z1[idx], z2[idx]
    N = z1.size(0)

    # [2N, d]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / tau  # cosine similarity divided by tau
    sim.masked_fill_(torch.eye(2 * N, device=z.device).bool(), -9e15)

    pos = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, pos)

# ---------- Visualisation ---------------------------------------------------------

def _tsne_plot(z, y, path):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    z_2d = TSNE(perplexity=30, n_iter=1_000).fit_transform(z)
    cmap = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    plt.figure(figsize=(6, 5))
    for lbl in range(-1, y.max().item() + 1):
        m = y == lbl
        col = "lightgray" if lbl == -1 else cmap[lbl]
        plt.scatter(z_2d[m, 0], z_2d[m, 1], s=6, c=col, label=str(lbl), alpha=0.6)
    plt.axis("off"); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

# ---------- One training epoch ----------------------------------------------------

@torch.no_grad()
def _maybe_viz(epoch, z, y, out_dir, every=10, n_viz=2000):
    """Optionally create a t-SNE plot every *every* epochs."""
    if y is None or epoch % every:
        return
    os.makedirs(out_dir, exist_ok=True)
    idx = torch.randperm(z.size(0))[:n_viz]
    _tsne_plot(z[idx].cpu().numpy(), y[idx].cpu(), f"{out_dir}/tsne_Ep{epoch:02d}.png")


def graphcl_epoch(
    model: torch.nn.Module,
    data: HeteroData,
    opt: torch.optim.Optimizer,
    device: torch.device,
    *,
    edge_drop: float = 0.3,
    feat_drop: float = 0.1,
    tau: float = 0.05,
    batch: int = 2048,
    y_label: torch.Tensor | None = None,
    viz_every: int = 10,
    viz_dir: str = "tsne",
):
    """Run **one** GraphCL training epoch and return the contrastive loss.

    If *y_label* is supplied, a t‑SNE plot is periodically generated.
    """
    model.train()

    # --- 1) Two graph views --------------------------------------------------
    d1 = data.clone().to(device)
    d2 = data.clone().to(device)

    d1["tweet"].x = mask_x(d1["tweet"].x, feat_drop)
    d2["tweet"].x = mask_x(d2["tweet"].x, feat_drop)

    for et in d1.edge_index_dict.keys():
        d1[et].edge_index, d1[et].edge_attr = drop_edge_and_attr(
            d1[et].edge_index, d1[et].edge_attr, edge_drop
        )
        d2[et].edge_index, d2[et].edge_attr = drop_edge_and_attr(
            d2[et].edge_index, d2[et].edge_attr, edge_drop
        )

    # --- 2) Forward pass -----------------------------------------------------
    _, z1, _ = model(d1.x_dict, d1.edge_index_dict, d1.edge_attr_dict)
    _, z2, _ = model(d2.x_dict, d2.edge_index_dict, d2.edge_attr_dict)

    loss = nt_xent(z1, z2, tau)

    # --- 3) Back‑propagation --------------------------------------------------
    opt.zero_grad(); loss.backward(); opt.step()

    # --- 4) Optional visualisation -------------------------------------------
    _maybe_viz(graphcl_epoch.ep, z1, y_label, viz_dir, viz_every)
    graphcl_epoch.ep += 1  # track current epoch as a static attribute
    return loss.item()

# Static attribute to keep track of the epoch counter
graphcl_epoch.ep = 1
