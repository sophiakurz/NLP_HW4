import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from model import BiHGT_Edge

def run_kfold(data, y, k, cfg, device="cuda", pretrained_path=None):
    # ---- 1) Prepare label indices -----------------------------------------
    y_cpu        = y.cpu()
    labeled_mask = y_cpu >= 0
    all_idx      = torch.arange(y_cpu.size(0))[labeled_mask]
    y_labeled    = y_cpu[labeled_mask]

    # ---- 2) Create Stratified K-Folds split -------------------------------
    skf     = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    # ---- 3) Read hyperparameters ------------------------------------------
    in_dim     = cfg['in_dim']
    hid        = cfg['hid']
    num_cls    = cfg['num_cls']
    lr         = float(cfg['lr'])
    epochs     = cfg['epochs']
    early_stop = cfg['early_stop']
    heads      = cfg.get('heads', 4)
    layers     = cfg.get('layers', 2)
    dropout    = cfg.get('dropout', 0.2)

    # ---- 4) Compute class weights -----------------------------------------
    class_freq = torch.bincount(y_cpu[labeled_mask], minlength=num_cls).float()
    class_freq[class_freq == 0] = 1.0
    class_w = (1.0 / class_freq)
    class_w = class_w / class_w.sum() * num_cls  # Normalize weights
    class_w = class_w.to(device)

    # ---- 5) Loop over folds -----------------------------------------------
    for fold, (train_loc, test_loc) in enumerate(
            skf.split(torch.zeros_like(y_labeled), y_labeled), 1):
        train_idx = all_idx[train_loc].to(device)
        test_idx  = all_idx[test_loc].to(device)

        # Instantiate model for this fold
        model = BiHGT_Edge(
            in_dim     = in_dim,
            hid_dim    = hid,
            out_dim    = num_cls,
            num_heads  = heads,
            num_layers = layers,
            dropout    = dropout,
        ).to(device)

        # Load pretrained weights, skipping the classification head
        if pretrained_path:
            state = torch.load(pretrained_path, map_location=device)
            state = {k: v for k, v in state.items() if not k.startswith('lin_out.')}
            model.load_state_dict(state, strict=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        best_macro, patience, best_state = 0.0, 0, None

        # ---- 6) Training loop ---------------------------------------------
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)[0]
            loss   = F.cross_entropy(logits[train_idx],
                                     y[train_idx].to(device),
                                     weight=class_w)
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)[0]
                val_pred   = val_logits[test_idx].argmax(dim=1).cpu()
                val_macro  = f1_score(y_cpu[test_idx.cpu()],
                                      val_pred,
                                      average='macro')

            # Early stopping on macro-F1
            if val_macro > best_macro:
                best_macro = val_macro
                patience   = 0
                best_state = model.state_dict()
            else:
                patience += 1
                if patience >= early_stop:
                    break

        # ---- 7) Load best model and evaluate --------------------------------
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_logits = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)[0]
            test_pred   = test_logits[test_idx].argmax(dim=1)
            acc         = (test_pred.cpu() == y_cpu[test_idx.cpu()]).float().mean().item()

        print(f"[Fold {fold}] Val F1 = {best_macro:.4f} | Test Acc = {acc:.4f}")
        results.append(acc)

    mean_acc = sum(results) / len(results)
    print(f"\n[ k-Fold Results ] accuracy per fold = {results} | mean = {mean_acc:.4f}")
    return results
