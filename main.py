import os, yaml, argparse, random, numpy as np, torch, traceback, tqdm
from torch_geometric.data import HeteroData

from build_graph      import build_hetero
from model            import BiHGT_Edge
from pretrain_graphcl import graphcl_epoch
from train_supervised import run_kfold

# ---------------- Utils ------------------------------------------------------
def fix_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

```python
# Count trainable parameters
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
```# -----------------------------------------------------------------------------

def main():
    # 1) CLI parsing
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    # 2) Load configuration and set up environment
    cfg = yaml.safe_load(open(args.config))
    print("[Setup] Loaded config:", cfg)
    fix_seed(cfg.get('seed', 42))
    device = torch.device(cfg.get('device', 'cpu'))
    print("[Setup] device:", device)

    work_dir = cfg['work_dir']; os.makedirs(work_dir, exist_ok=True)

    # 3) Graph construction (if not present)
    graph_pt = cfg['graph_pt']
    if not os.path.exists(graph_pt):
        build_hetero(
            raw_dir   = cfg['raw_dir'],
            save_path = graph_pt,
            device    = device.type,
            max_k     = cfg['edge_k'],
            keep_bu   = cfg['keep_bu']
        )

    # 4) Load the heterogeneous graph
    bundle: dict      = torch.load(graph_pt, map_location=device, weights_only=False)
    data : HeteroData = bundle['data'].to(device)
    y = data['tweet'].y                       # [N] (can be used for t-SNE)
    feat_dim  = data['tweet'].x.size(1)
    num_class = int(y[y>=0].max().item()+1)

    print(f"[Data] nodes={data['tweet'].num_nodes}, feat_dim={feat_dim}")
    print(f"[Data] labeled={(y>=0).sum().item()}, classes={num_class}")

    # 5) Hyper-parameters
    mcfg, gcfg, scfg = cfg['model'], cfg['graphcl'], cfg['supervised']

    # 6) GraphCL pre-training ---------------------------------------------------
    print("\n[Pretrain] GraphCL …")
    model = BiHGT_Edge(
        in_dim     = feat_dim,
        hid_dim    = mcfg['hid'],
        out_dim    = num_class,
        num_heads  = mcfg['heads'],
        num_layers = mcfg['layers'],
        dropout    = mcfg.get('dropout', .2)
    ).to(device)
    print(f" params={count_params(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=float(gcfg['lr']))
    warm_steps = 10
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=gcfg['epochs']-warm_steps
    )

    viz_dir = os.path.join(work_dir, "viz")            # Directory to save t-SNE visualizations
    tau_val = gcfg.get('tau', 0.05)                     # default 0.05
    for ep in range(1, gcfg['epochs']+1):
        loss = graphcl_epoch(
            model, data, opt, device,
            edge_drop = gcfg.get('edge_drop', .3),
            feat_drop = gcfg.get('feat_drop', .1),
            tau       = tau_val,
            y_label   = y,                # pass labels for coloring
            viz_every = 10,               # visualize every 10 epochs
            viz_dir   = viz_dir
        )
        if ep > warm_steps: scheduler.step()
        lr_now = opt.param_groups[0]['lr']
        tqdm.tqdm.write(f"[GraphCL] Ep{ep:02d} lr={lr_now:.5f} loss={loss:.4f}")

    pre_path = os.path.join(work_dir, 'pretrain.pt')
    torch.save(model.state_dict(), pre_path)
    print(f"[✓] saved {pre_path}")

    # 7) k-fold fine-tuning -----------------------------------------------------
    print("\n[Finetune] k-fold …")
    run_kfold(
        data, y, k=scfg['k_folds'],
        cfg=dict(
            in_dim     = feat_dim,
            hid        = mcfg['hid'],
            num_cls    = num_class,
            lr         = scfg['lr'],
            epochs     = scfg['epochs'],
            early_stop = scfg['early_stop'],
            heads      = mcfg['heads'],
            layers     = mcfg['layers'],
            dropout    = mcfg.get('dropout', .2),
        ),
        device=device,
        pretrained_path=pre_path
    )

    print("\n[Done] all finished.")

if __name__ == '__main__':
    main()
