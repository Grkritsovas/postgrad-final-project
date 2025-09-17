import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict

# model wrapper for pretrained models
class PretrainedModel(nn.Module):
    """Load pretrained models from torchvision."""
    
    def __init__(self, backbone_name='efficientnet_b0', freeze=True, dropout=0.5):
        super().__init__()
        
        # Channel adapter for grayscale to RGB
        self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1)
        
        # Load pretrained backbone
        if backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            self.backbone.classifier = nn.Identity()
            feature_dim = 1280
        elif backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Identity()
            feature_dim = 512
        elif backbone_name == 'mobilenet_v3':
            self.backbone = models.mobilenet_v3_small(pretrained=True)
            self.backbone.classifier = nn.Identity()
            feature_dim = 576
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.channel_adapter(x)
        features = self.backbone(x)
        return self.regressor(features)

def evaluate_music2emo(embedding_dir: Path, labels_df, test_ids):
    """Zero-shot evaluation of music2emo on DEAM labels."""
    import sys, numpy as np, torch
    from sklearn.metrics import mean_squared_error, r2_score
    from huggingface_hub import snapshot_download, hf_hub_download
    from tqdm import tqdm

    # load repo code
    REPO_ID = "amaai-lab/music2emo"
    repo_local = snapshot_download(REPO_ID)
    sys.path.append(repo_local)

    from model.linear_mt_attn_ck import FeedforwardModelMTAttnCK

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FeedforwardModelMTAttnCK(
        input_size=768*2, # 1536-d MERT (CLS+pool)
        output_size_classification=56,
        output_size_regression=2
    ).to(device)

    # checkpoint
    ckpt_path = hf_hub_download(REPO_ID, filename="saved_models/D_all.ckpt")
    # NOTE: avoid weights_only kwarg for broad torch compatibility
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    preds = {}
    ranges_probe = [] # to infer output scale

    with torch.no_grad():
        for sid in tqdm(test_ids, desc="music2emo inference"):
            p = embedding_dir / f"{int(sid)}.npy"
            if not p.exists():
                continue
            emb = np.load(p).astype(np.float32)
            if emb.ndim != 1 or emb.shape[0] != 1536:
                continue
            x = torch.from_numpy(emb).unsqueeze(0).to(device) # (1,1536)
            pad = torch.zeros((1, 1), dtype=torch.long, device=device)
            model_input = {
                "x_mert": x,
                "x_chord_root": pad,
                "x_chord_attr": pad,
                "x_key": pad
            }
            _, reg_out = model(model_input) # (1,2)
            y = reg_out.squeeze(0).detach().cpu().numpy()
            preds[int(sid)] = [float(y[0]), float(y[1])]
            ranges_probe.append(y)

    # infer scale: if |pred| <= ~1.5, assume [-1,1] and map to 1..9
    if ranges_probe:
        arr = np.vstack(ranges_probe)
        needs_denorm = np.nanmax(np.abs(arr)) <= 1.5
    else:
        needs_denorm = False

    def _to_deam_scale(v):
        return v * 4.0 + 5.0 if needs_denorm else v

    y_true_v, y_pred_v, y_true_a, y_pred_a = [], [], [], []
    for sid, (pv, pa) in preds.items():
        if sid in labels_df.index:
            y_true_v.append(float(labels_df.loc[sid, "valence_mean"]))
            y_true_a.append(float(labels_df.loc[sid, "arousal_mean"]))
            y_pred_v.append(_to_deam_scale(pv))
            y_pred_a.append(_to_deam_scale(pa))

    if len(y_true_v) == 0:
        return {"rmse_v": float("nan"), "rmse_a": float("nan"),
                "r2_v": float("nan"), "r2_a": float("nan")}

    return {
        "rmse_v": float(np.sqrt(mean_squared_error(y_true_v, y_pred_v))),
        "rmse_a": float(np.sqrt(mean_squared_error(y_true_a, y_pred_a))),
        "r2_v":   float(r2_score(y_true_v, y_pred_v)),
        "r2_a":   float(r2_score(y_true_a, y_pred_a)),
    }

def save_results(name: str, history: Dict, test_metrics: Dict, save_dir: Path):
    """Save experiment results to JSON."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'name': name,
        'history': history,
        'test_metrics': test_metrics
    }
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(save_dir / f'{name}.json', 'w') as f:
        json.dump(convert(results), f, indent=2)