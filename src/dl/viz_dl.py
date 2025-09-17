from pathlib import Path
import json, torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 300

# load histories from .ckpt
def load_histories_from_ckpts(named_paths: dict):
    """named_paths: {label: path_to_ckpt}
    returns: {label: history_dict}"""
    out = {}
    for label, p in named_paths.items():
        p = Path(p)
        if not p.exists():
            print(f"[skip] {label}: {p} not found"); continue
        state = torch.load(p, map_location="cpu")
        hist = state.get("history", {})
        if not hist:
            print(f"[skip] {label}: no 'history' in ckpt"); continue
        out[label] = hist
    return out

# load test metrics from JSONs
def load_results_from_jsons(named_paths: dict):
    """named_paths: {label: path_to_test_metrics_json}
    returns: {label: metric_dict}"""
    out = {}
    for label, p in named_paths.items():
        p = Path(p)
        if not p.exists():
            print(f"[skip] {label}: {p} not found"); continue
        with open(p, "r") as f:
            d = json.load(f)
        # support either nested {'rmse_v':...} or {'test_metrics':{...}}
        metrics = d.get("test_metrics", d)
        out[label] = {
            "Valence RMSE": float(metrics["rmse_v"]),
            "Arousal RMSE": float(metrics["rmse_a"]),
            "Valence R²":   float(metrics["r2_v"]),
            "Arousal R²":   float(metrics["r2_a"]),
        }
    return out

# plots
def plot_histories_val_rmse(histories: dict, savename: Path|None=None, title="Validation RMSE (Valence)"):
    plt.figure(figsize=(8.5,5.5))
    for label, hist in histories.items():
        y = hist.get("val_rmse_v")
        if y:
            plt.plot(y, label=label)
    plt.xlabel("Epoch"); plt.ylabel("RMSE (valence)"); plt.title(title)
    plt.grid(alpha=0.4, linestyle="--"); plt.legend()
    plt.tight_layout()
    if savename:
        Path(savename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savename, bbox_inches="tight")
        print(f"saved: {savename}")
    plt.show()

def plot_results_bar(results: dict, metric_keys=("Valence RMSE","Arousal RMSE","Valence R²","Arousal R²"),
                     savename: Path|None=None, title="Final Test Metrics"):
    df = pd.DataFrame(results).T  # rows=labels, cols=metrics
    df = df[list(metric_keys)]
    ax = df.plot(kind="bar", figsize=(10,6), rot=0, width=0.82)
    plt.title(title); plt.ylabel("Score"); plt.xlabel("Model")
    plt.grid(axis="y", linestyle="--", alpha=0.5); plt.tight_layout()
    # bar labels
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", padding=2, fontsize=8)
    if savename:
        Path(savename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savename, bbox_inches="tight")
        print(f"saved: {savename}")
    plt.show()