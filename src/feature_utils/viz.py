import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_descriptor_importance(importance_report: dict):
    # pool importance by stat suffix (…_mean, …_std, etc.)
    bag = {}
    for base, info in importance_report.items():
        for col, score in info['importance_scores'].items():
            stat = col.rsplit('_', 1)[-1]
            bag.setdefault(stat, []).append(score)
    stat_means = {k: float(np.mean(v)) for k, v in bag.items()}
    items = sorted(stat_means.items(), key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(8,5))
    plt.bar([k for k,_ in items], [v for _,v in items])
    plt.title("Average Importance of Statistical Descriptors")
    plt.ylabel("Mean RF importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_mid_level_correlations(mid_df: pd.DataFrame):
    corr = mid_df.corr()
    plt.figure(figsize=(9,7))
    im = plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Mid-Level Feature Correlations")
    plt.tight_layout()
    plt.show()