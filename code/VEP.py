#!/usr/bin/env python3
"""
Comprehensive comparison of 7B vs 40B model performance across trait categories.
Includes DeLong tests, bootstrap CIs, FDR correction, and visualizations.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================

# Statistical parameters
FDR_ALPHA = 0.05              # False Discovery Rate threshold for significance
BOOTSTRAP_ITERATIONS = 1000   # Number of bootstrap resamples for confidence intervals
BOOTSTRAP_CI = 0.95           # Confidence interval level (0.95 = 95% CI)
RANDOM_SEED = 42              # Random seed for reproducibility

# Visualization parameters
N_EXAMPLE_CATEGORIES = 4      # Number of categories to show in ROC/PR curves
FIGURE_DPI = 300              # Resolution for saved figures

# ============================================================================

def delong_test(y_true, y_score1, y_score2):
    """
    DeLong test for comparing two ROC curves.
    Returns p-value for the hypothesis that AUC1 = AUC2.
    
    Based on DeLong et al. (1988) "Comparing the areas under two or more 
    correlated receiver operating characteristic curves: a nonparametric approach"
    """
    n = len(y_true)
    
    # Compute AUCs
    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)
    
    # Get indices of positive and negative samples
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    if n_pos == 0 or n_neg == 0:
        return np.nan
    
    # Compute structural components for both classifiers
    def compute_structural_components(scores, pos_idx, neg_idx):
        V10 = np.zeros(len(pos_idx))
        V01 = np.zeros(len(neg_idx))
        
        for i, pos in enumerate(pos_idx):
            V10[i] = np.mean(scores[pos] > scores[neg_idx]) + 0.5 * np.mean(scores[pos] == scores[neg_idx])
        
        for i, neg in enumerate(neg_idx):
            V01[i] = np.mean(scores[pos_idx] > scores[neg]) + 0.5 * np.mean(scores[pos_idx] == scores[neg])
        
        return V10, V01
    
    V10_1, V01_1 = compute_structural_components(y_score1, pos_idx, neg_idx)
    V10_2, V01_2 = compute_structural_components(y_score2, pos_idx, neg_idx)
    
    # Compute covariance
    S10_1 = np.var(V10_1, ddof=1)
    S01_1 = np.var(V01_1, ddof=1)
    S10_2 = np.var(V10_2, ddof=1)
    S01_2 = np.var(V01_2, ddof=1)
    
    S10_12 = np.cov(V10_1, V10_2, ddof=1)[0, 1]
    S01_12 = np.cov(V01_1, V01_2, ddof=1)[0, 1]
    
    # Compute variance of difference
    var_diff = (S10_1 / n_pos + S01_1 / n_neg + 
                S10_2 / n_pos + S01_2 / n_neg - 
                2 * S10_12 / n_pos - 2 * S01_12 / n_neg)
    
    if var_diff <= 0:
        return np.nan
    
    # Compute z-score and p-value
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return p_value


def bootstrap_metric_ci(y_true, y_score1, y_score2, metric_func, n_iterations=None, ci=None):
    """
    Bootstrap confidence interval for the difference in metrics between two models.
    Note: Random seed should be set globally, not inside this function.
    
    Parameters:
    -----------
    n_iterations : int, optional
        Number of bootstrap iterations (defaults to BOOTSTRAP_ITERATIONS global)
    ci : float, optional
        Confidence interval level (defaults to BOOTSTRAP_CI global)
    """
    if n_iterations is None:
        n_iterations = BOOTSTRAP_ITERATIONS
    if ci is None:
        ci = BOOTSTRAP_CI
    
    n_samples = len(y_true)
    differences = []
    
    # DO NOT reset seed here - it makes bootstrap CIs artificially similar across categories
    for _ in range(n_iterations):
        # Sample with replacement
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true[idx]
        score1_boot = y_score1[idx]
        score2_boot = y_score2[idx]
        
        # Skip if no positive or negative samples
        if len(np.unique(y_boot)) < 2:
            continue
        
        try:
            m1 = metric_func(y_boot, score1_boot)
            m2 = metric_func(y_boot, score2_boot)
            differences.append(m2 - m1)
        except:
            continue
    
    if len(differences) == 0:
        return np.nan, np.nan
    
    # Compute percentiles
    alpha = 1 - ci
    lower = np.percentile(differences, 100 * alpha / 2)
    upper = np.percentile(differences, 100 * (1 - alpha / 2))
    
    return lower, upper


def benjamini_hochberg_fdr(p_values, alpha=None):
    """
    Benjamini-Hochberg FDR correction.
    Returns adjusted q-values.
    Handles NaN p-values by keeping them as NaN in output.
    
    Parameters:
    -----------
    alpha : float, optional
        FDR significance threshold (defaults to FDR_ALPHA global)
    """
    if alpha is None:
        alpha = FDR_ALPHA
    
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Identify valid (non-NaN) p-values
    valid_mask = ~np.isnan(p_values)
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        return np.full(n, np.nan)
    
    # Initialize q-values as NaN
    q_values = np.full(n, np.nan)
    
    # Get valid p-values and their original indices
    valid_indices = np.where(valid_mask)[0]
    valid_p = p_values[valid_mask]
    
    # Sort valid p-values and keep track of original indices
    sorted_idx = np.argsort(valid_p)
    sorted_p = valid_p[sorted_idx]
    
    # Compute q-values for valid p-values only
    q_valid = np.zeros(n_valid)
    prev_q = 1.0
    
    for i in range(n_valid - 1, -1, -1):
        rank = i + 1
        q = min(prev_q, sorted_p[i] * n_valid / rank)
        q_valid[sorted_idx[i]] = q
        prev_q = q
    
    # Put q-values back in original positions
    q_values[valid_indices] = q_valid
    
    return q_values


def analyze_trait_category(df, category, model1_col='score_7b', model2_col='score_40b'):
    """
    Analyze a single trait category and return all metrics.
    """
    df_cat = df[df['category'] == category].copy()
    
    if len(df_cat) < 10:  # Skip categories with too few samples
        return None
    
    y_true = df_cat['label'].values
    
    # Skip if only one class present
    if len(np.unique(y_true)) < 2:
        return None
    
    y_score1 = df_cat[model1_col].values
    y_score2 = df_cat[model2_col].values
    
    # Compute metrics
    try:
        auroc_7b = roc_auc_score(y_true, y_score1)
        auroc_40b = roc_auc_score(y_true, y_score2)
        auprc_7b = average_precision_score(y_true, y_score1)
        auprc_40b = average_precision_score(y_true, y_score2)
    except:
        return None
    
    delta_auroc = auroc_40b - auroc_7b
    delta_auprc = auprc_40b - auprc_7b
    
    # DeLong test
    p_delong = delong_test(y_true, y_score1, y_score2)
    
    # Bootstrap CI for ΔAUROC
    ci_lower, ci_upper = bootstrap_metric_ci(y_true, y_score1, y_score2, roc_auc_score)
    
    # Sample size info
    n_total = len(df_cat)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    return {
        'category': category,
        'n_total': n_total,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'auroc_7b': auroc_7b,
        'auroc_40b': auroc_40b,
        'auprc_7b': auprc_7b,
        'auprc_40b': auprc_40b,
        'delta_auroc': delta_auroc,
        'delta_auprc': delta_auprc,
        'p_delong': p_delong,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def merge_datasets(file_7b, file_40b, dataset_name):
    """
    Merge 7B and 40B datasets for a given dataset (ClinVar or IDH).
    Verifies row-by-row alignment before using row-based matching.
    """
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name} datasets...")
    print(f"{'='*60}")
    
    # Load datasets
    df_7b = pd.read_csv(file_7b)
    df_40b = pd.read_csv(file_40b)
    
    print(f"7B dataset shape: {df_7b.shape}")
    print(f"40B dataset shape: {df_40b.shape}")
    
    merge_cols = ['chrom', 'pos', 'ref', 'alt']
    
    # Check if datasets are aligned (same number of rows AND matching coordinates)
    if len(df_7b) == len(df_40b):
        # Verify row-by-row alignment
        df_7b_sorted = df_7b.reset_index(drop=True)
        df_40b_sorted = df_40b.reset_index(drop=True)
        
        coords_match = (
            (df_7b_sorted['chrom'].values == df_40b_sorted['chrom'].values).all() and
            (df_7b_sorted['pos'].values == df_40b_sorted['pos'].values).all() and
            (df_7b_sorted['ref'].values == df_40b_sorted['ref'].values).all() and
            (df_7b_sorted['alt'].values == df_40b_sorted['alt'].values).all()
        )
        
        if coords_match:
            print(f"\n✓ Datasets are perfectly aligned (verified row-by-row)")
            
            # Use row-based matching
            df_merged = pd.DataFrame({
                'chrom': df_7b_sorted['chrom'],
                'pos': df_7b_sorted['pos'],
                'ref': df_7b_sorted['ref'],
                'alt': df_7b_sorted['alt'],
                'label': df_7b_sorted['label'],
                'category': df_7b_sorted['category'],
                'score_7b': -df_7b_sorted['llr'],  # NEGATIVE LLR
                'score_40b': -df_40b_sorted['llr']  # NEGATIVE LLR
            })
        else:
            print(f"\n⚠ Same row count but coordinates don't match - using coordinate-based merge")
            
            # Fall back to coordinate-based merge
            df_merged = df_7b[merge_cols + ['label', 'category', 'llr']].merge(
                df_40b[merge_cols + ['llr']],
                on=merge_cols,
                suffixes=('_7b', '_40b'),
                how='inner'
            )
            # RENAME FIRST, then negate
            df_merged = df_merged.rename(columns={'llr_7b': 'score_7b', 'llr_40b': 'score_40b'})
            df_merged['score_7b'] = -df_merged['score_7b']  # NEGATIVE LLR
            df_merged['score_40b'] = -df_merged['score_40b']  # NEGATIVE LLR
    else:
        # Different row counts - use coordinate-based merge
        print(f"\n⚠ Different row counts - using coordinate-based merge")
        
        df_merged = df_7b[merge_cols + ['label', 'category', 'llr']].merge(
            df_40b[merge_cols + ['llr']],
            on=merge_cols,
            suffixes=('_7b', '_40b'),
            how='inner'
        )
        # RENAME FIRST, then negate
        df_merged = df_merged.rename(columns={'llr_7b': 'score_7b', 'llr_40b': 'score_40b'})
        df_merged['score_7b'] = -df_merged['score_7b']  # NEGATIVE LLR
        df_merged['score_40b'] = -df_merged['score_40b']  # NEGATIVE LLR
    
    print(f"\nMerged dataset shape: {df_merged.shape}")
    print(f"Categories: {df_merged['category'].nunique()}")
    print(f"Category distribution:\n{df_merged['category'].value_counts()}")
    
    # Filter out Unclassified category
    if 'Unclassified' in df_merged['category'].values:
        n_unclassified = len(df_merged[df_merged['category'] == 'Unclassified'])
        df_merged = df_merged[df_merged['category'] != 'Unclassified']
        print(f"\n✓ Removed Unclassified category (n={n_unclassified} variants)")
        print(f"Final dataset shape: {df_merged.shape}")
        print(f"Final categories: {df_merged['category'].nunique()}")
        print(f"Final distribution:\n{df_merged['category'].value_counts()}")
    
    print(f"\n✓ Using NEGATIVE LLR scores (-llr) for analysis")
    
    return df_merged

def create_forest_plot(results_df, dataset_name, output_file):
    """
    Create a forest plot showing ΔAUROC with 95% CI for each trait category.
    ENHANCED: Larger fonts and exact ΔAUROC values displayed as percentages on the right side.
    Uses global FDR_ALPHA for significance threshold.
    """
    # Sort by delta_auroc
    results_df = results_df.sort_values('delta_auroc', ascending=True)
    
    # Create figure with extra width for value labels
    fig, ax = plt.subplots(figsize=(14, max(6, len(results_df) * 0.4)))
    
    # Plot each point individually to control colors
    y_pos = np.arange(len(results_df))
    
    for i, row in enumerate(results_df.itertuples()):
        color = 'red' if row.q_fdr < FDR_ALPHA else 'gray'
        xerr_lower = row.delta_auroc - row.ci_lower
        xerr_upper = row.ci_upper - row.delta_auroc
        
        ax.errorbar(row.delta_auroc, i, 
                    xerr=[[xerr_lower], [xerr_upper]],
                    fmt='o', color='black', ecolor=color, elinewidth=2.5, 
                    capsize=5, capthick=2, markersize=8)
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Get x-axis limits for positioning text
    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    
    # Add exact ΔAUROC values as percentages on the right side
    for i, row in enumerate(results_df.itertuples()):
        color = 'red' if row.q_fdr < FDR_ALPHA else 'gray'
        # Position text to the right of the upper CI bound
        x_pos = row.ci_upper + 0.03 * x_range
        ax.text(x_pos, i, f'{row.delta_auroc*100:.2f}%', 
                fontsize=11, va='center', ha='left', 
                color=color,
                fontweight='bold')
    
    # Format axes with larger fonts
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['category'], fontsize=13)
    ax.set_xlabel('ΔAUROC (40B - 7B)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Trait Category', fontsize=15, fontweight='bold')
    ax.set_title(f'{dataset_name}: Model Performance Improvement\n(Red = FDR q < {FDR_ALPHA})', 
                 fontsize=17, fontweight='bold', pad=20)
    
    # Update tick label sizes
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=13)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    # Add sample sizes as text on the left
    for i, row in enumerate(results_df.itertuples()):
        ax.text(xlim[0] - 0.08 * x_range, i, f'n={row.n_total}', 
                fontsize=10, va='center', ha='right', color='gray')
    
    # Adjust x-limits to ensure text fits on both sides
    ax.set_xlim(xlim[0] - 0.12 * x_range, xlim[1] + 0.18 * x_range)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved forest plot to: {output_file}")
    
    return fig

def create_roc_pr_curves(df, results_df, dataset_name, output_file, n_examples=None):
    """
    Create ROC and PR curves for selected trait categories.
    Uses global N_EXAMPLE_CATEGORIES and FIGURE_DPI.
    """
    if n_examples is None:
        n_examples = N_EXAMPLE_CATEGORIES
    
    # Select categories: best gain, near-zero, worst (if any)
    results_sorted = results_df.sort_values('delta_auroc', ascending=False)
    
    selected = []
    
    # Best improvement
    if len(results_sorted) > 0:
        best_cat = results_sorted.iloc[0]['category']
        selected.append(best_cat)
    
    # Near-zero change (avoid duplicates)
    near_zero = results_sorted.iloc[(results_sorted['delta_auroc'].abs()).argsort()[:1]]
    if len(near_zero) > 0:
        near_zero_cat = near_zero.iloc[0]['category']
        if near_zero_cat not in selected:
            selected.append(near_zero_cat)
    
    # Worst (negative if exists) - avoid duplicates
    if len(results_sorted) > 2:
        worst_cat = results_sorted.iloc[-1]['category']
        if worst_cat not in selected:
            selected.append(worst_cat)
    
    # One more middle category if available - avoid duplicates
    if len(results_sorted) > 3 and len(selected) < n_examples:
        mid_idx = len(results_sorted) // 2
        mid_cat = results_sorted.iloc[mid_idx]['category']
        if mid_cat not in selected:
            selected.append(mid_cat)
    
    # If we still need more categories and have them available
    remaining_idx = 1
    while len(selected) < n_examples and len(selected) < len(results_sorted):
        if remaining_idx >= len(results_sorted):
            break
        candidate = results_sorted.iloc[remaining_idx]['category']
        if candidate not in selected:
            selected.append(candidate)
        remaining_idx += 1
    
    selected = selected[:n_examples]
    
    # Create subplots
    n_cats = len(selected)
    fig, axes = plt.subplots(2, n_cats, figsize=(5*n_cats, 8))
    
    if n_cats == 1:
        axes = axes.reshape(2, 1)
    
    for idx, category in enumerate(selected):
        df_cat = df[df['category'] == category]
        y_true = df_cat['label'].values
        y_score_7b = df_cat['score_7b'].values
        y_score_40b = df_cat['score_40b'].values
        
        # Get metrics
        cat_results = results_df[results_df['category'] == category].iloc[0]
        
        # ROC curve
        ax_roc = axes[0, idx]
        fpr_7b, tpr_7b, _ = roc_curve(y_true, y_score_7b)
        fpr_40b, tpr_40b, _ = roc_curve(y_true, y_score_40b)
        
        ax_roc.plot(fpr_7b, tpr_7b, label=f'7B (AUC={cat_results.auroc_7b:.3f})', 
                   linewidth=2, color='blue')
        ax_roc.plot(fpr_40b, tpr_40b, label=f'40B (AUC={cat_results.auroc_40b:.3f})', 
                   linewidth=2, color='red')
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        ax_roc.set_xlabel('False Positive Rate', fontsize=10)
        ax_roc.set_ylabel('True Positive Rate', fontsize=10)
        ax_roc.set_title(f'{category}\nΔAUROC={cat_results.delta_auroc:.4f}', 
                        fontsize=10, fontweight='bold')
        ax_roc.legend(loc='lower right', fontsize=8)
        ax_roc.grid(alpha=0.3)
        
        # PR curve
        ax_pr = axes[1, idx]
        prec_7b, rec_7b, _ = precision_recall_curve(y_true, y_score_7b)
        prec_40b, rec_40b, _ = precision_recall_curve(y_true, y_score_40b)
        
        ax_pr.plot(rec_7b, prec_7b, label=f'7B (AP={cat_results.auprc_7b:.3f})', 
                  linewidth=2, color='blue')
        ax_pr.plot(rec_40b, prec_40b, label=f'40B (AP={cat_results.auprc_40b:.3f})', 
                  linewidth=2, color='red')
        
        # Baseline (proportion of positives)
        baseline = np.mean(y_true)
        ax_pr.axhline(y=baseline, color='k', linestyle='--', alpha=0.3)
        
        ax_pr.set_xlabel('Recall', fontsize=10)
        ax_pr.set_ylabel('Precision', fontsize=10)
        ax_pr.set_title(f'ΔAUPRC={cat_results.delta_auprc:.4f}', fontsize=10)
        ax_pr.legend(loc='best', fontsize=8)
        ax_pr.grid(alpha=0.3)
    
    plt.suptitle(f'{dataset_name}: Representative ROC and PR Curves', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved ROC/PR curves to: {output_file}")
    
    return fig


def analyze_dataset(file_7b, file_40b, dataset_name):
    """
    Complete analysis pipeline for a dataset.
    """
    print(f"\n{'#'*60}")
    print(f"# ANALYZING: {dataset_name}")
    print(f"{'#'*60}")
    
    # Step 0: Merge datasets
    df = merge_datasets(file_7b, file_40b, dataset_name)
    
    # Step 1-3: Analyze each trait category
    print(f"\nAnalyzing trait categories...")
    results = []
    categories = df['category'].unique()
    
    for category in categories:
        result = analyze_trait_category(df, category)
        if result is not None:
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Step 4: FDR correction
    print(f"\nApplying Benjamini-Hochberg FDR correction...")
    results_df['q_fdr'] = benjamini_hochberg_fdr(results_df['p_delong'].values)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY: {dataset_name}")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    
    # Save results
    output_csv = f'outputs/{dataset_name.lower().replace(" ", "_")}_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to: {output_csv}")
    
    # Step 5: Create plots
    print(f"\nGenerating visualizations...")
    
    # Forest plot
    forest_file = f'outputs/{dataset_name.lower().replace(" ", "_")}_forest_plot.png'
    create_forest_plot(results_df, dataset_name, forest_file)
    
    # ROC/PR curves
    curves_file = f'outputs/{dataset_name.lower().replace(" ", "_")}_roc_pr_curves.png'
    create_roc_pr_curves(df, results_df, dataset_name, curves_file)
    
    print(f"\n{'='*60}")
    print(f"SIGNIFICANCE SUMMARY: {dataset_name}")
    print(f"{'='*60}")
    sig_cats = results_df[results_df['q_fdr'] < FDR_ALPHA]
    print(f"Categories with significant improvement (q < {FDR_ALPHA}): {len(sig_cats)}/{len(results_df)}")
    if len(sig_cats) > 0:
        print("\nSignificant categories:")
        print(sig_cats[['category', 'delta_auroc', 'p_delong', 'q_fdr']].to_string(index=False))
    
    return df, results_df


# Main execution
if __name__ == "__main__":
    # Set random seed once globally for reproducibility
    # This ensures bootstrap CIs are reproducible but NOT artificially similar across categories
    np.random.seed(RANDOM_SEED)
    
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON ANALYSIS")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - FDR alpha: {FDR_ALPHA}")
    print(f"  - Bootstrap iterations: {BOOTSTRAP_ITERATIONS}")
    print(f"  - Bootstrap CI: {BOOTSTRAP_CI*100:.0f}%")
    print(f"  - Random seed: {RANDOM_SEED}")
    print(f"  - Figure DPI: {FIGURE_DPI}")
    print(f"  - Example categories in plots: {N_EXAMPLE_CATEGORIES}")
    print(f"{'='*60}")
    print(f"{'='*60}")
    
    # Analyze ClinVar dataset
    df_clinvar, results_clinvar = analyze_dataset(
        'data/VEP/clinvar_snv_subset_10k_with_scores_7b.csv',
        'data/VEP/clinvar_snv_subset_10k_with_scores_40b.csv',
        'ClinVar'
    )
    
    # Analyze IDH dataset
    df_idh, results_idh = analyze_dataset(
        'data/VEP/idh_subset_with_scores_7b.csv',
        'data/VEP/idh_subset_with_scores_40b.csv',
        'IDH'
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nAll results and visualizations have been saved to outputs/")