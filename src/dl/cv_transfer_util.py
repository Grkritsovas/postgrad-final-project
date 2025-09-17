from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from torchvision import models

# Label scaling
def norm_labels(v: np.ndarray, a: np.ndarray, dataset: str) -> np.ndarray:
    """Return labels normalized to [-1,1]"""
    if dataset == "deam":  # 1..9
        return np.stack([(v - 5.0) / 4.0, (a - 5.0) / 4.0], axis=-1).astype(np.float32)
    elif dataset == "deezer":  # -3..3
        return np.stack([v / 3.0, a / 3.0], axis=-1).astype(np.float32)
    else:
        raise ValueError("dataset must be 'deam' or 'deezer'")

def denorm_labels(y: torch.Tensor, dataset: str) -> torch.Tensor:
    """Map predictions back to original label scale"""
    if dataset == "deam":
        return y * 4.0 + 5.0
    elif dataset == "deezer":
        return y * 3.0
    else:
        raise ValueError("dataset must be 'deam' or 'deezer'")

# Dataset
class SpectrogramDataset(Dataset):
    """
    Loads precomputed spectrogram segments from <spec_dir>/<song_id>.npz with 'spectrograms' array
    Returns (tensor[1,H,W], tensor[2], song_id)
    """
    def __init__(
        self,
        song_ids: List[int],
        spec_dir: Path,
        labels_df: pd.DataFrame,
        dataset: str, # 'deam' or 'deezer'
        max_segments_per_song: int = 4,
        require_exists: bool = True,
    ):
        self.spec_dir = Path(spec_dir)
        self.labels_df = labels_df
        self.dataset = dataset
        self.max_segments = max_segments_per_song

        self.samples: List[Tuple[int, int, Path]] = [] # (song_id, segment_idx, path)
        missing = 0
        for sid in song_ids:
            p = self.spec_dir / f"{int(sid)}.npz"
            if not p.exists():
                missing += 1
                if require_exists:
                    continue
            else:
                try:
                    n = np.load(p)["spectrograms"].shape[0]
                except Exception:
                    continue
                for seg in range(min(n, self.max_segments)):
                    self.samples.append((int(sid), seg, p))
        if missing and require_exists:
            print(f"Warning: {missing} spectrogram files missing in {spec_dir}")
        print(f"Dataset: {len(self.samples)} segments from {len(song_ids)} songs [{self.dataset}]")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sid, seg, p = self.samples[idx]
        arr = np.load(p)["spectrograms"][seg] # (H,W)
        x = torch.from_numpy(arr).float().unsqueeze(0) # (1,H,W)

        if self.dataset == "deam":
            v = float(self.labels_df.loc[sid, "valence_mean"])
            a = float(self.labels_df.loc[sid, "arousal_mean"])
        else:
            row = self.labels_df.loc[self.labels_df["dzr_sng_id"] == sid].iloc[0]
            v, a = float(row["valence"]), float(row["arousal"])

        y = torch.from_numpy(norm_labels(np.array([v]), np.array([a]), self.dataset)[0])  # (2,)
        return x, y, sid

# Backbone selection
def default_backbone_for(model_type: str) -> str:
    """
    - panns, vggish, clap -> efficientnet_b0
    - musicnn -> mobilenet_v3_small
    - ast -> mobilenet_v3_small (tie-breaker on valence)
    """
    model_type = model_type.lower()
    if model_type in ["panns", "vggish", "clap"]:
        return "efficientnet_b0"
    if model_type in ["musicnn", "ast"]:
        return "mobilenet_v3_small"
    raise ValueError("model_type must be one of: ast, clap, musicnn, panns, vggish")

def build_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    name = name.lower()
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(pretrained=pretrained)
        m.classifier = nn.Identity()
        feat_dim = 1280
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(pretrained=pretrained)
        m.classifier = nn.Identity()
        feat_dim = 576
    else:
        raise ValueError("backbone must be 'efficientnet_b0' or 'mobilenet_v3_small'")
    return m, feat_dim

# BN helpers
def _is_bn(m):
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))

def set_bn_eval(module: nn.Module):
    """Force BN layers to eval-mode so running stats are not updated."""
    for m in module.modules():
        if _is_bn(m):
            m.eval() # stops running_mean/var updates

def freeze_bn_affine(module: nn.Module):
    """Disable gradients for BN affine params (gamma/beta)"""
    for m in module.modules():
        if _is_bn(m) and m.affine:
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)

def freeze_bn_in_last_blocks(model: "CVRegressor", k: int = 1, freeze_affine: bool = True):
    """Freeze BN (eval + no grads) inside the last k backbone blocks"""
    blocks = _backbone_blocks(model.backbone_name, model.backbone)
    for blk in blocks[-k:]:
        set_bn_eval(blk)
        if freeze_affine:
            freeze_bn_affine(blk)

def extract_head_state(model: "CVRegressor") -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    return {k: v for k, v in sd.items() if k.startswith(("adapter.", "regressor."))}

def load_head_state(model: "CVRegressor", state: Dict[str, torch.Tensor]):
    model.load_state_dict(state, strict=False)  # ignores backbone keys

# Model

class CVRegressor(nn.Module):
    """
    1-channel spectrogram -> 3-channel adapter -> CV backbone -> regressor (Tanh to [-1,1]).
    Resize to input_size for stability across backbones.
    """
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        freeze_backbone: bool = True,
        dropout: float = 0.3,
        input_size: Tuple[int, int] = (224, 224)
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.input_size = input_size
        self.adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained=True)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
            nn.Tanh()
        )

    def forward(self, x): # x: (B,1,H,W)
        x = self.adapter(x)
        if self.input_size is not None:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        feats = self.backbone(x) # pooled & flattened vector for both backbones
        return self.regressor(feats)

# Metrics (song-level)

@torch.no_grad()
def song_level_metrics(
    preds: torch.Tensor, labels: torch.Tensor, song_ids: List[int], dataset: str
) -> Dict[str, float]:
    # aggregate by song (mean over segments)
    by_song: Dict[int, Dict[str, List[torch.Tensor]]] = {}
    for p, y, sid in zip(preds, labels, song_ids):
        d = by_song.setdefault(int(sid), {"p": [], "y": y})
        d["p"].append(p)
    P, Y = [], []
    for _, d in by_song.items():
        P.append(torch.stack(d["p"]).mean(0))
        Y.append(d["y"])
    P = torch.stack(P)
    Y = torch.stack(Y)

    # back to original scale
    P_orig = denorm_labels(P, dataset).cpu().numpy()
    Y_orig = denorm_labels(Y, dataset).cpu().numpy()

    rmse_v = float(np.sqrt(mean_squared_error(Y_orig[:, 0], P_orig[:, 0])))
    rmse_a = float(np.sqrt(mean_squared_error(Y_orig[:, 1], P_orig[:, 1])))
    r2_v = float(r2_score(Y_orig[:, 0], P_orig[:, 0]))
    r2_a = float(r2_score(Y_orig[:, 1], P_orig[:, 1]))
    return {"rmse_v": rmse_v, "rmse_a": rmse_a, "r2_v": r2_v, "r2_a": r2_a}

# Trainer
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        es_patience: int = 10,
         *,
        freeze_bn: bool = False,
        bn_scope: str = "backbone", # "backbone" or "all"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="min", factor=0.5, patience=5, verbose=True)
        self.crit = nn.MSELoss(reduction="mean")
        self.grad_clip = grad_clip
        self.es_patience = es_patience
        self.history = {"train_loss": [], "val_loss": [], "val_rmse_v": [], "val_rmse_a": [], "val_r2_v": [], "val_r2_a": []}
        self.freeze_bn = freeze_bn
        self.bn_scope = bn_scope

    def _step_batch(self, batch, training: bool):
        x, y, _ = batch
        x, y = x.to(self.device), y.to(self.device)
        if training:
            self.model.train()
            # keep BN frozen (no stat updates) while training
            if self.freeze_bn:
                target = self.model.backbone if self.bn_scope == "backbone" else self.model
                set_bn_eval(target)
            pred = self.model(x)
            loss = self.crit(pred, y)
            self.opt.zero_grad()
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            return loss.item(), pred.detach(), y.detach()
        else:
            self.model.eval()
            with torch.no_grad():
                pred = self.model(x)
                loss = self.crit(pred, y)
            return loss.item(), pred, y

    def train_epoch(self, loader: DataLoader) -> float:
        running = 0.0
        for batch in loader:
            loss, _, _ = self._step_batch(batch, training=True)
            running += loss
        return running / max(1, len(loader))

    @torch.no_grad()
    def validate(self, loader: DataLoader, dataset: str) -> Dict[str, float]:
        losses, all_p, all_y, all_sid = [], [], [], []
        self.model.eval()
        for x, y, sid in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            losses.append(self.crit(pred, y).item())
            all_p.append(pred.cpu()); all_y.append(y.cpu()); all_sid += list(sid)
        P = torch.cat(all_p)
        Y = torch.cat(all_y)
        m = song_level_metrics(P, Y, all_sid, dataset)
        return {"loss": float(np.mean(losses)), **m}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset: str,
        epochs: int,
        ckpt_path: Optional[Path] = None
    ) -> Dict[str, List[float]]:
        best_loss = float("inf")
        best_epoch = 0
        best_vals: Optional[Dict[str, float]] = None
        patience = 0
        for ep in range(1, epochs + 1):
            tr = self.train_epoch(train_loader)
            va = self.validate(val_loader, dataset)
            self.sched.step(va["loss"])

            self.history["train_loss"].append(tr)
            self.history["val_loss"].append(va["loss"])
            self.history["val_rmse_v"].append(va["rmse_v"])
            self.history["val_rmse_a"].append(va["rmse_a"])
            self.history["val_r2_v"].append(va["r2_v"])
            self.history["val_r2_a"].append(va["r2_a"])

            if va["loss"] < best_loss:
                best_loss, best_epoch, best_vals = va["loss"], ep, va
                patience = 0
                if ckpt_path:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"model": self.model.state_dict(),
                         "opt": self.opt.state_dict(),
                         "history": self.history},
                        ckpt_path
                    )
            else:
                patience += 1
                if patience >= self.es_patience:
                    break

        if best_vals is not None:
            print(f"Best epoch {best_epoch} | "
                  f"val_loss={best_vals['loss']:.4f} | "
                  f"RMSE V/A={best_vals['rmse_v']:.3f}/{best_vals['rmse_a']:.3f} | "
                  f"R2 V/A={best_vals['r2_v']:.3f}/{best_vals['r2_a']:.3f}")
        return self.history

    @torch.no_grad()
    def test(self, loader: DataLoader, dataset: str) -> Dict[str, float]:
        return self.validate(loader, dataset)

# DataLoader helpers

def make_loaders(
    spec_dir: Path,
    labels_df: pd.DataFrame,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: Optional[List[int]] = None,
    dataset: str = 'deam',
    batch_size: int = 32,
    max_segments_per_song: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True
):
    train_ds = SpectrogramDataset(train_ids, spec_dir, labels_df, dataset, max_segments_per_song)
    val_ds   = SpectrogramDataset(val_ids,   spec_dir, labels_df, dataset, max_segments_per_song)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dl = None
    if test_ids is not None:
        test_ds = SpectrogramDataset(test_ids, spec_dir, labels_df, dataset, max_segments_per_song)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_dl, val_dl, test_dl

# High-level pipelines

def train_deam_only(
    model_type: str, # 'ast' / 'clap' / 'musicnn' / 'panns' /'vggish'
    backbone: Optional[str], # None -> default_backbone_for(model_type)
    deam_spec_dir: Path,
    deam_labels: pd.DataFrame, # index: song_id, cols: valence_mean/arousal_mean
    train_ids: List[int], val_ids: List[int], test_ids: List[int],
    *,
    freeze_backbone: bool = True,
    input_size: Tuple[int, int] = (224, 224),
    lr: float = 1e-4, weight_decay: float = 1e-5,
    batch_size: int = 32, epochs: int = 50, patience: int = 10,
    max_segments_per_song: int = 4,
    ckpt_path: Optional[Path] = None
) -> Tuple[nn.Module, Dict[str,List[float]], Dict[str,float]]:
    backbone = backbone or default_backbone_for(model_type)
    print(f"DEAM-only | model_type={model_type} | backbone={backbone} | freeze_backbone={freeze_backbone}")

    train_dl, val_dl, test_dl = make_loaders(
        deam_spec_dir, deam_labels, train_ids, val_ids, test_ids,
        dataset="deam", batch_size=batch_size, max_segments_per_song=max_segments_per_song
    )

    model = CVRegressor(backbone, freeze_backbone=freeze_backbone, input_size=input_size)
    trainer = Trainer(model, lr=lr, weight_decay=weight_decay, es_patience=patience)
    history = trainer.fit(train_dl, val_dl, dataset="deam", epochs=epochs, ckpt_path=ckpt_path)
    test_metrics = trainer.test(test_dl, dataset="deam")
    print(f"TEST RMSE(V/A) {test_metrics['rmse_v']:.3f}/{test_metrics['rmse_a']:.3f} | R2(V/A) {test_metrics['r2_v']:.3f}/{test_metrics['r2_a']:.3f}")
    if ckpt_path is not None:
        save_test_metrics_json(
            ckpt_path.with_name("test_metrics.json"),
            test_metrics,
            extra={"model_type": model_type, "backbone": backbone, "dataset": "DEAM"}
        )
    return model, history, test_metrics

def deezer_pretrain_then_deam_finetune(
    model_type: str,
    backbone: Optional[str],
    deezer_spec_dir: Path,
    deezer_df: pd.DataFrame, # columns: dzr_sng_id, valence, arousal
    deezer_train_ids: List[int], deezer_val_ids: List[int],
    deam_spec_dir: Path,
    deam_labels: pd.DataFrame, # index: song_id
    deam_train_ids: List[int], deam_val_ids: List[int], deam_test_ids: List[int],
    *,
    freeze_backbone_finetune: bool = True,
    input_size: Tuple[int, int] = (224, 224),
    pre_epochs: int = 15, ft_epochs: int = 30,
    pre_lr: float = 1e-4, ft_lr: float = 1e-6,
    weight_decay: float = 1e-5,
    batch_size_pre: int = 32, batch_size_ft: int = 32,
    max_segments_per_song: int = 4,
    pre_ckpt: Optional[Path] = None, ft_ckpt: Optional[Path] = None
) -> Tuple[nn.Module, Dict[str,List[float]], Dict[str,float]]:
    backbone = backbone or default_backbone_for(model_type)
    print(f"Pretrain(FULL) on Deezer → Finetune on DEAM | model_type={model_type} | backbone={backbone}")

    # Stage 1: Deezer pretraining (head only)
    pre_train_dl, pre_val_dl, _ = make_loaders(
        deezer_spec_dir, deezer_df, deezer_train_ids, deezer_val_ids, None,
        dataset="deezer", batch_size=batch_size_pre, max_segments_per_song=max_segments_per_song
    )

    # freeze backbone so only adapter+regressor learn
    pre_model = CVRegressor(backbone, freeze_backbone=True, input_size=input_size)

    pre_trainer = Trainer(pre_model, lr=pre_lr, weight_decay=weight_decay, es_patience=3)
    pre_hist = pre_trainer.fit(pre_train_dl, pre_val_dl, dataset="deezer", epochs=pre_epochs, ckpt_path=None)

    # save only head weights to pre_ckpt (discard Deezer BN stats)
    if pre_ckpt is not None:
        pre_ckpt.parent.mkdir(parents=True, exist_ok=True)
        head_state = extract_head_state(pre_model)
        torch.save({"head": head_state, "history": pre_hist}, pre_ckpt)

    # Stage 2: DEAM finetuning (usually freeze backbone + tiny LR on head)
    ft_train_dl, ft_val_dl, ft_test_dl = make_loaders(
        deam_spec_dir, deam_labels, deam_train_ids, deam_val_ids, deam_test_ids,
        dataset="deam", batch_size=batch_size_ft, max_segments_per_song=max_segments_per_song
    )

    # fresh ImageNet backbone- keep frozen for simplicity/stability
    ft_model = CVRegressor(backbone, freeze_backbone=freeze_backbone_finetune, input_size=input_size)

    # load head-only weights from pretraining (if present)
    if pre_ckpt and Path(pre_ckpt).exists():
        state = torch.load(pre_ckpt, map_location="cpu")
        load_head_state(ft_model, state.get("head", {}))
        print(f"Loaded head-only weights from {pre_ckpt}")

    # train on DEAM (head only, since backbone is frozen)
    ft_trainer = Trainer(ft_model, lr=ft_lr, weight_decay=weight_decay, es_patience=5)
    ft_hist = ft_trainer.fit(ft_train_dl, ft_val_dl, dataset="deam", epochs=ft_epochs, ckpt_path=ft_ckpt)
    test_metrics = ft_trainer.test(ft_test_dl, dataset="deam")

    print(f"TEST RMSE(V/A) {test_metrics['rmse_v']:.3f}/{test_metrics['rmse_a']:.3f} | R2(V/A) {test_metrics['r2_v']:.3f}/{test_metrics['r2_a']:.3f}")

    if ft_ckpt is not None:
        save_test_metrics_json(
            ft_ckpt.with_name("test_metrics_with_pretraining.json"),
            test_metrics,
            extra={"model_type": model_type, "backbone": backbone, "dataset": "DEAM", "pretrained_on": "Deezer"}
        )

    # merge histories with prefixes
    merged_hist = {f"pre_{k}": v for k, v in pre_hist.items()}
    for k, v in ft_hist.items():
        merged_hist[f"ft_{k}"] = v
    return ft_model, merged_hist, test_metrics

# freeze helpers
def _backbone_blocks(backbone_name: str, backbone: nn.Module):
    """Return a list of top-level conv blocks for supported torchvision backbones."""
    name = backbone_name.lower()
    if "efficientnet" in name:
        # torchvision EfficientNet has .features as Sequential of blocks
        return list(backbone.features)
    if "mobilenet_v3" in name:
        # torchvision MobileNetV3 has .features as Sequential of blocks
        return list(backbone.features)
    raise ValueError(f"Unsupported backbone for partial unfreeze: {backbone_name}")

def freeze_backbone_all(backbone: nn.Module):
    for p in backbone.parameters():
        p.requires_grad = False

def unfreeze_last_backbone_blocks(model: "CVRegressor", k: int = 2):
    """Freeze all backbone params, then unfreeze the last k blocks."""
    freeze_backbone_all(model.backbone)
    blocks = _backbone_blocks(model.backbone_name, model.backbone)
    for blk in blocks[-k:]:
        for p in blk.parameters():
            p.requires_grad = True

def save_test_metrics_json(path: Path, metrics: Dict[str, float], extra: Optional[Dict[str, str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: float(v) for k, v in metrics.items()}
    if extra:
        out.update(extra)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True