from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

# Label scaling helpers
def _norm_labels(v: np.ndarray, a: np.ndarray, dataset: str) -> np.ndarray:
    if dataset == "deam":   # 1..9 ->[-1,1]
        return np.stack([(v - 5.0)/4.0, (a - 5.0)/4.0], axis=-1).astype(np.float32)
    elif dataset == "deezer":  # [-3,3] ->[-1,1]
        return np.stack([v/3.0, a/3.0], axis=-1).astype(np.float32)
    raise ValueError("dataset must be 'deam' or 'deezer'")

def _denorm_labels(y: torch.Tensor, dataset: str) -> torch.Tensor:
    if dataset == "deam":   # [-1,1] -> 1..9
        return y * 4.0 + 5.0
    elif dataset == "deezer":  # [-1,1] -> [-3,3]
        return y * 3.0
    raise ValueError("dataset must be 'deam' or 'deezer'")

# Dataset + loaders

class MertEmbedDataset(Dataset):
    """
    Expects one embedding per song: <emb_dir>/<song_id>.npy  (shape: (1536,))
    DEAM: labels_df indexed by song_id with ['valence_mean','arousal_mean']
    Deezer: labels_df has columns ['dzr_sng_id','valence','arousal'].
    """
    def __init__(self,
                 ids: List[int],
                 labels_df: pd.DataFrame,
                 emb_dir: Path,
                 dataset: str = "deam",
                 require_exists: bool = True,
                 in_dim: int = 1536):
        self.ids_all = [int(x) for x in ids]
        self.labels_df = labels_df
        self.emb_dir = Path(emb_dir)
        self.dataset = dataset
        self.in_dim = in_dim

        self.ids = []
        missing = 0
        for sid in self.ids_all:
            if (self.emb_dir / f"{sid}.npy").exists():
                self.ids.append(sid)
            else:
                missing += 1
                if not require_exists:
                    self.ids.append(sid)
        if missing:
            print(f"[MERT] Missing {missing}/{len(self.ids_all)} .npy in {emb_dir} (using {len(self.ids)})")

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        emb = np.load(self.emb_dir / f"{sid}.npy").astype(np.float32)  # (in_dim,)
        if emb.ndim != 1 or emb.shape[0] != self.in_dim:
            raise ValueError(f"Embedding for {sid} has shape {emb.shape}, expected ({self.in_dim},)")

        if self.dataset == "deam":
            v = float(self.labels_df.loc[sid, "valence_mean"])
            a = float(self.labels_df.loc[sid, "arousal_mean"])
        else:
            row = self.labels_df.loc[self.labels_df["dzr_sng_id"] == sid].iloc[0]
            v, a = float(row["valence"]), float(row["arousal"])

        y = _norm_labels(np.array([v]), np.array([a]), self.dataset)[0]  # (2,)
        return torch.from_numpy(emb), torch.from_numpy(y), sid

def make_mert_loaders(
    emb_dir: Path,
    labels_df: pd.DataFrame,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: Optional[List[int]] = None,
    dataset: str = "deam",
    batch_size: int = 32,
    in_dim: int = 1536,
    num_workers: int = 2,
    pin_memory: bool = True
):
    train_ds = MertEmbedDataset(train_ids, labels_df, emb_dir, dataset, True, in_dim)
    val_ds   = MertEmbedDataset(val_ids,   labels_df, emb_dir, dataset, True, in_dim)
    test_ds  = MertEmbedDataset(test_ids,  labels_df, emb_dir, dataset, True, in_dim) if test_ids is not None else None

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dl  = None
    if test_ds is not None:
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_dl, val_dl, test_dl


# Model (tiny MLP head)

class VARegressor(nn.Module):
    def __init__(self, in_dim: int = 1536, hidden: int = 256, drop: float = 0.3, use_bn: bool = True):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden)]
        if use_bn: layers += [nn.BatchNorm1d(hidden)]
        layers += [nn.ReLU(inplace=True), nn.Dropout(drop), nn.Linear(hidden, 128), nn.ReLU(inplace=True),
                   nn.Dropout(drop), nn.Linear(128, 2), nn.Tanh()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x): # x: (B, in_dim)
        return self.mlp(x)


# Metrics

@torch.no_grad()
def metrics(pred: torch.Tensor, true: torch.Tensor, dataset: str) -> Dict[str, float]:
    # Denorm for interpretable metrics
    P = _denorm_labels(pred, dataset).cpu().numpy()
    Y = _denorm_labels(true, dataset).cpu().numpy()
    rmse_v = float(np.sqrt(mean_squared_error(Y[:,0], P[:,0])))
    rmse_a = float(np.sqrt(mean_squared_error(Y[:,1], P[:,1])))
    r2_v   = float(r2_score(Y[:,0], P[:,0]))
    r2_a   = float(r2_score(Y[:,1], P[:,1]))
    return {"rmse_v": rmse_v, "rmse_a": rmse_a, "r2_v": r2_v, "r2_a": r2_a}

# Trainer

class MertTrainer:
    def __init__(self, model: nn.Module, lr=1e-4, weight_decay=1e-5, es_patience=10, grad_clip: float = 0.75):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="min", factor=0.5, patience=5, verbose=False)
        self.crit = nn.MSELoss()
        self.es_patience = es_patience
        self.grad_clip = grad_clip
        self.history = {"train_loss": [], "val_loss": [], "val_rmse_v": [], "val_rmse_a": [], "val_r2_v": [], "val_r2_a": []}

    def _epoch(self, dl: DataLoader, training: bool) -> Tuple[float, torch.Tensor, torch.Tensor]:
        total, n = 0.0, 0
        preds, trues = [], []
        self.model.train(mode=training)
        for xb, yb, _ in dl:
            xb, yb = xb.to(self.device), yb.to(self.device)
            with torch.set_grad_enabled(training):
                out = self.model(xb)
                loss = self.crit(out, yb)
                if training:
                    self.opt.zero_grad()
                    loss.backward()
                    if self.grad_clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.opt.step()
            total += loss.item() * len(xb); n += len(xb)
            preds.append(out.detach()); trues.append(yb.detach())
        return total/max(1,n), torch.cat(preds), torch.cat(trues)

    def fit(self, train_dl: DataLoader, val_dl: DataLoader, dataset: str, epochs: int, ckpt_path: Optional[Path]=None):
        best, best_ep, best_val_tuple = float("inf"), 0, None
        patience = 0
        for ep in range(1, epochs+1):
            tr_loss, _, _ = self._epoch(train_dl, True)
            va_loss, vp, vy = self._epoch(val_dl, False)
            m = metrics(vp, vy, dataset)
            self.history["val_rmse_v"].append(m["rmse_v"])
            self.history["val_rmse_a"].append(m["rmse_a"])
            self.history["val_r2_v"].append(m["r2_v"])
            self.history["val_r2_a"].append(m["r2_a"])
            self.sched.step(va_loss)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)

            if va_loss < best:
                best, best_ep = va_loss, ep
                best_val_tuple = (vp, vy)
                patience = 0
                if ckpt_path:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": self.model.state_dict(),
                                "opt": self.opt.state_dict(),
                                "history": self.history}, ckpt_path)
            else:
                patience += 1
                if patience >= self.es_patience:
                    break

        assert best_val_tuple is not None
        vp, vy = best_val_tuple
        m = metrics(vp, vy, dataset)
        print(f"Best epoch {best_ep} | val_loss={best:.4f} | RMSE V/A={m['rmse_v']:.3f}/{m['rmse_a']:.3f} | R2 V/A={m['r2_v']:.3f}/{m['r2_a']:.3f}")
        return self.history

    @torch.no_grad()
    def test(self, dl: DataLoader, dataset: str) -> Dict[str, float]:
        self.model.eval()
        preds, trues = [], []
        for xb, yb, _ in dl:
            out = self.model(xb.to(self.device))
            preds.append(out.cpu()); trues.append(yb.cpu())
        P = torch.cat(preds); Y = torch.cat(trues)
        return metrics(P, Y, dataset)

# Save test JSON

def save_test_json(path: Path, metrics: Dict[str,float], extra: Optional[Dict[str,str]]=None, suffix: str=""):
    path = Path(path)
    if suffix:
        path = path.with_name(path.stem + f"_{suffix}" + path.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: float(v) for k, v in metrics.items()}
    if extra: out.update(extra)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

# Pipelines
def mert_deam_only(
    emb_dir: Path,
    labels_df: pd.DataFrame, # index: song_id
    train_ids: List[int], val_ids: List[int], test_ids: List[int],
    *,
    in_dim: int = 1536,
    lr: float = 1e-4, weight_decay: float = 1e-5,
    batch_size: int = 32, epochs: int = 80, patience: int = 10,
    hidden: int = 256, drop: float = 0.3, use_bn: bool = True,
    ckpt_path: Optional[Path] = None, metrics_path: Optional[Path] = None
):
    print("MERT → DEAM-only")
    train_dl, val_dl, test_dl = make_mert_loaders(
        emb_dir, labels_df, train_ids, val_ids, test_ids,
        dataset="deam", batch_size=batch_size, in_dim=in_dim
    )
    model = VARegressor(in_dim=in_dim, hidden=hidden, drop=drop, use_bn=use_bn)
    tr = MertTrainer(model, lr=lr, weight_decay=weight_decay, es_patience=patience)
    history = tr.fit(train_dl, val_dl, dataset="deam", epochs=epochs, ckpt_path=ckpt_path)
    test_metrics = tr.test(test_dl, dataset="deam")
    print(f"TEST RMSE(V/A) {test_metrics['rmse_v']:.3f}/{test_metrics['rmse_a']:.3f} | R2(V/A) {test_metrics['r2_v']:.3f}/{test_metrics['r2_a']:.3f}")
    if metrics_path:
        save_test_json(metrics_path, test_metrics, extra={"dataset":"DEAM","model":"MERT-MLP"}, suffix="deam_only")
    return model, history, test_metrics

def mert_deezer_pretrain_then_deam_finetune(
    deezer_emb_dir: Path, deezer_df: pd.DataFrame, deezer_train_ids: List[int], deezer_val_ids: List[int],
    deam_emb_dir: Path,    deam_labels: pd.DataFrame, deam_train_ids: List[int], deam_val_ids: List[int], deam_test_ids: List[int],
    *,
    in_dim: int = 1536,
    pre_epochs: int = 20, pre_lr: float = 1e-4,
    ft_epochs: int = 50,  ft_lr: float = 7e-6,
    weight_decay: float = 1e-5,
    batch_size_pre: int = 64, batch_size_ft: int = 32,
    hidden: int = 256, drop: float = 0.3, use_bn: bool = True,
    pre_ckpt: Optional[Path] = None, ft_ckpt: Optional[Path] = None, metrics_path: Optional[Path] = None
):
    print("Pretrain on Deezer MERT → Finetune on DEAM MERT")
    # Stage 1: Deezer
    pre_train_dl, pre_val_dl, _ = make_mert_loaders(
        deezer_emb_dir, deezer_df, deezer_train_ids, deezer_val_ids, None,
        dataset="deezer", batch_size=batch_size_pre, in_dim=in_dim
    )
    pre_model = VARegressor(in_dim=in_dim, hidden=hidden, drop=drop, use_bn=use_bn)
    pre_tr = MertTrainer(pre_model, lr=pre_lr, weight_decay=weight_decay, es_patience=3)
    pre_hist = pre_tr.fit(pre_train_dl, pre_val_dl, dataset="deezer", epochs=pre_epochs, ckpt_path=pre_ckpt)

    # Stage 2: DEAM
    ft_train_dl, ft_val_dl, ft_test_dl = make_mert_loaders(
        deam_emb_dir, deam_labels, deam_train_ids, deam_val_ids, deam_test_ids,
        dataset="deam", batch_size=batch_size_ft, in_dim=in_dim
    )
    ft_model = VARegressor(in_dim=in_dim, hidden=hidden, drop=drop, use_bn=use_bn)
    if pre_ckpt and Path(pre_ckpt).exists():
        state = torch.load(pre_ckpt, map_location="cpu")
        ft_model.load_state_dict(state["model"], strict=False)
        print(f"Loaded pretrained head from {pre_ckpt}")

    ft_tr = MertTrainer(ft_model, lr=ft_lr, weight_decay=weight_decay, es_patience=5)
    ft_hist = ft_tr.fit(ft_train_dl, ft_val_dl, dataset="deam", epochs=ft_epochs, ckpt_path=ft_ckpt)
    test_metrics = ft_tr.test(ft_test_dl, dataset="deam")
    print(f"TEST RMSE(V/A) {test_metrics['rmse_v']:.3f}/{test_metrics['rmse_a']:.3f} | R2(V/A) {test_metrics['r2_v']:.3f}/{test_metrics['r2_a']:.3f}")

    if metrics_path:
        save_test_json(metrics_path, test_metrics,
                       extra={"dataset":"DEAM","model":"MERT-MLP","pretrained_on":"Deezer"},
                       suffix="with_pretraining")

    merged_hist = {f"pre_{k}": v for k, v in pre_hist.items()}
    for k, v in ft_hist.items(): merged_hist[f"ft_{k}"] = v
    return ft_model, merged_hist, test_metrics

def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True