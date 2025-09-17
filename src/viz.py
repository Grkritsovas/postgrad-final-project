from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from src.utils import desc_of

def plot_descriptor_importance(importance_report: dict):
    # pool importance by stat suffix (_mean, _std, etc.)
    bag = {}
    for base, info in importance_report.items():
        for col, score in info['importance_scores'].items():
            stat = desc_of(col)
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

def plot_combined_histories(histories: Dict, metric: str = 'val_loss', filename: str = None):

    plt.figure(figsize=(10, 7))
    
    for name, history in histories.items():
        if metric in history:
            plt.plot(history[metric], label=name)
            
    plt.title(f'Comparison of Models: {metric.replace("_", " ").title()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if filename:
        output_dir = Path('../results/plots/transfer_learning/')
        output_dir.mkdir(parents=True, exist_ok=True)
        full_path = output_dir / filename
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {full_path}")
        
    plt.show()

def plot_results_comparison(results: dict, filename: str = None):
    """
    Creates a bar plot to compare final model metrics and optionally saves it.
    results (dict): A nested dictionary of final metrics for each model.
    filename (str, optional): The name of the output file (e.g., 'comparison.png').
    """
    # Convert results to a pandas DataFrame for easy plotting
    results_df = pd.DataFrame(results).T # .T transposes for correct orientation
    
    ax = results_df.plot(
        kind='bar', 
        figsize=(12, 7), 
        rot=0, # Keep model names horizontal
        width=0.8
    )
    
    plt.title('Final Model Performance Comparison')
    plt.ylabel('Metric Score')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metrics')
    
    # Add value labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9, padding=3)

    plt.ylim(0, max(results_df.max()) * 1.15) # some space at top
    plt.tight_layout()

    if filename:
        output_dir = Path('../results/plots/transfer_learning/')
        output_dir.mkdir(parents=True, exist_ok=True)
        full_path = output_dir / filename
        # Save figure
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        
        print(f"Plot saved to {full_path}")
        
    plt.show()

def load_and_pair_results(results_dir):
    """Load all results and pair original vs custom splits"""
    
    results_dir = Path(results_dir)
    all_files = list(results_dir.glob("*.json"))
    
    # Separate by split type
    orig_results = {}
    custom_results = {}
    
    for file in all_files:
        with open(file, 'r') as f:
            data = json.load(f)
            
        name = data['name']
        metrics = data['test_metrics']
        
        # Identify the model and method
        if 'Orig_Split' in name:
            # Extract base name without _Orig_Split
            base_name = name.replace('_Orig_Split', '')
            orig_results[base_name] = metrics
        else:
            # Custom split (no suffix or different naming)
            if 'Custom_Split' in name:
                base_name = name.replace('_Custom_Split', '')
            custom_results[base_name] = metrics
    
    return orig_results, custom_results

def visualize_split_comparison(orig_results, custom_results):
    """Create comparison visualization."""
    
    # Find common models
    common_models = set(orig_results.keys()) & set(custom_results.keys())
    
    if not common_models:
        print("No matching models found between splits")
        print(f"Original split models: {list(orig_results.keys())}")
        print(f"Custom split models: {list(custom_results.keys())}")
        return
    
    models = sorted(list(common_models))
    
    # Extract metrics
    orig_rmse_v = [orig_results[m]['rmse_v'] for m in models]
    custom_rmse_v = [custom_results[m]['rmse_v'] for m in models]
    orig_rmse_a = [orig_results[m]['rmse_a'] for m in models]
    custom_rmse_a = [custom_results[m]['rmse_a'] for m in models]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Valence subplot
    bars1 = ax1.bar(x - width/2, orig_rmse_v, width, label='Original (3.6%)', color='coral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, custom_rmse_v, width, label='Custom (15%)', color='teal', alpha=0.8)
    
    # Add degradation percentage on top
    for i in range(len(models)):
        sign = '+' if orig_rmse_v[i] > custom_rmse_v[i] else '-'
        degradation = ((orig_rmse_v[i] - custom_rmse_v[i]) / custom_rmse_v[i]) * 100
        y_pos = max(orig_rmse_v[i], custom_rmse_v[i]) + 0.05
        ax1.text(i, y_pos, f'{sign}{degradation:.0f}%', ha='center', fontsize=9)
    
    ax1.set_ylabel('RMSE')
    ax1.set_title('Valence RMSE - Test Set Size Impact')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[:20] for m in models], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Arousal subplot  
    bars3 = ax2.bar(x - width/2, orig_rmse_a, width, label='Original (3.6%)', color='coral', alpha=0.8)
    bars4 = ax2.bar(x + width/2, custom_rmse_a, width, label='Custom (15%)', color='teal', alpha=0.8)
    
    # Add degradation percentage
    for i in range(len(models)):
        sign = '+' if orig_rmse_a[i] > custom_rmse_a[i] else '-'
        degradation = ((orig_rmse_a[i] - custom_rmse_a[i]) / custom_rmse_a[i]) * 100
        y_pos = max(orig_rmse_a[i], custom_rmse_a[i]) + 0.05
        ax2.text(i, y_pos, f'{sign}{degradation:.0f}%', ha='center', fontsize=9)
    
    ax2.set_ylabel('RMSE')
    ax2.set_title('Arousal RMSE - Test Set Size Impact')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:20] for m in models], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Impact of Test Set Size: 58 samples (3.6%) vs 270 samples (15%)', fontsize=14)
    plt.tight_layout()
    plt.savefig('../results/test_split_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate average degradation
    avg_deg_v = np.mean([(o-c)/c*100 for o,c in zip(orig_rmse_v, custom_rmse_v)])
    avg_deg_a = np.mean([(o-c)/c*100 for o,c in zip(orig_rmse_a, custom_rmse_a)])
    
    print(f"\nAverage performance degradation with 58-sample test set:")
    print(f"  Valence: {avg_deg_v:.1f}% worse")
    print(f"  Arousal: {avg_deg_a:.1f}% worse")
    
    # Print detailed table
    print(f"\n{'Model':<30} {'V_orig':<8} {'V_custom':<8} {'V_diff%':<8} {'A_orig':<8} {'A_custom':<8} {'A_diff%':<8}")
    print("-" * 90)
    for i, m in enumerate(models):
        v_diff = ((orig_rmse_v[i] - custom_rmse_v[i])/custom_rmse_v[i])*100
        a_diff = ((orig_rmse_a[i] - custom_rmse_a[i])/custom_rmse_a[i])*100
        print(f"{m[:30]:<30} {orig_rmse_v[i]:<8.3f} {custom_rmse_v[i]:<8.3f} {v_diff:<8.1f} "
              f"{orig_rmse_a[i]:<8.3f} {custom_rmse_a[i]:<8.3f} {a_diff:<8.1f}")
        
#-------Visualization Functions for main pipeline
#Visualizes the stability of category importance across different data splits.
def plot_category_mass_heatmap(category_sums_per_fold: dict[int, pd.Series],
                               zscore_within_fold: bool = True,
                               title: str = "Category total SHAP mass across folds"):
    cats  = sorted(set().union(*[set(s.index) for s in category_sums_per_fold.values()]))
    folds = sorted(category_sums_per_fold.keys())
    M = pd.DataFrame(0.0, index=cats, columns=folds)
    for f, s in category_sums_per_fold.items():
        M.loc[s.index, f] = s.values
    if zscore_within_fold:
        M = (M - M.mean(axis=0)) / (M.std(axis=0, ddof=0) + 1e-9)
        center = 0.0; cmap = "vlag"
    else:
        center = None; cmap = "YlOrRd"
    plt.figure(figsize=(6, 3 + 0.3*len(cats)))
    sns.heatmap(M, annot=False, cmap=cmap, center=center)
    plt.title(title); plt.xlabel("Fold"); plt.ylabel("Perceptual category")
    plt.tight_layout(); plt.show()

# visualize the output of k_sweep_weighting_compare_nested
def plot_weighting_k_curves(df_unw, df_w, title="Impact of weighting across k"):
    plt.figure(figsize=(7,4))
    plt.errorbar(df_unw.k, df_unw.rmse_mean, yerr=df_unw.rmse_std, fmt='-o', label='Train unweighted')
    plt.errorbar(df_w.k,   df_w.rmse_mean,   yerr=df_w.rmse_std,   fmt='-s', label='Train weighted')
    plt.legend(); plt.grid(True)
    plt.xlabel("k"); plt.ylabel("CV RMSE"); plt.title(title)
    plt.tight_layout(); plt.show()

import sys
sys.path.append('..')
from src.utils import evaluate_descriptor_for_category
from src.make_dataset.aggregate import DEFAULT_8, FULL

def visualize_category_selections(
    category_selections: Dict[str, List[str]],
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame,
    descriptor_set: str = "DEFAULT_8"
):
    """
    Left:  0/1 selection of descriptor types per perceptual category
    Right: performance score of each descriptor type per category
    Columns = descriptor types from src.aggregate.{DEFAULT_8 | FULL}
    Rows    = perceptual categories from hierarchy_map
    """
    # choose descriptor set
    if descriptor_set.upper() == "FULL":
        desc_types = list(FULL)
    else:
        desc_types = list(DEFAULT_8)

    annot_performance = (descriptor_set.upper() != "FULL")

    categories = sorted(hierarchy_map['perceptual'].dropna().unique().tolist())

    # Selection matrix (0/1)
    selection_matrix = pd.DataFrame(0, index=categories, columns=desc_types, dtype=int)
    for cat, descs in category_selections.items():
        if cat not in selection_matrix.index:
            continue
        for d in descs:
            dt = desc_of(d) # normalize to descriptor suffix
            if dt in selection_matrix.columns:
                selection_matrix.loc[cat, dt] = 1

    # Performance matrix
    performance_matrix = pd.DataFrame(index=categories, columns=desc_types, dtype=float)
    for category in categories:
        bases = hierarchy_map[hierarchy_map['perceptual'] == category]['feature'].tolist()
        for dt in desc_types:
            score = evaluate_descriptor_for_category(agg_df, y, category, bases, dt)
            performance_matrix.loc[category, dt] = float(score)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(selection_matrix, cmap='RdBu_r', center=0.5, annot=True,
                fmt='g', cbar_kws={'label': 'Selected'}, ax=axes[0])
    axes[0].set_title('Selected Descriptor Types by Category')
    axes[0].set_xlabel('Descriptor type'); axes[0].set_ylabel('Perceptual category')

    sns.heatmap(performance_matrix.astype(float),
                cmap='YlOrRd',
                annot=annot_performance,
                fmt='.3f' if annot_performance else '',
                cbar_kws={'label': 'Score'},
                ax=axes[1])
    axes[1].set_title('Descriptor-Type Performance by Category')
    axes[1].set_xlabel('Descriptor type'); axes[1].set_ylabel('Perceptual category')

    # tidy labels when FULL (many columns)
    if descriptor_set.upper() == "FULL":
        for ax in axes:
            ax.tick_params(axis='x', labelrotation=45, labelsize=9)
            ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout(); plt.show()
    return selection_matrix, performance_matrix