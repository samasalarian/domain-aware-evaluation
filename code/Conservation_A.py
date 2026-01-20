"""
TraitGym Statistical Analysis: Spearman Correlation + Permutation Testing + FDR
Analyzes the relationship between PhastCons conservation scores and Evo2 LLR scores
across different trait groups.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import logit
from statsmodels.stats.multitest import multipletests
from typing import Tuple, Dict
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================

MIN_GROUP_SIZE = 15  # Minimum variants per trait group
N_PERMUTATIONS = 10000  # Number of permutations for testing
N_BOOTSTRAP = 1000  # Number of bootstrap iterations for CI
FDR_ALPHA = 0.1  # FDR significance threshold
N_BINS = 10  # Number of bins for monotonicity plot

# Input files
FILE_7B = 'data/Conservation/traitgym_complex_phastcon_llr_scores_7b.csv'
FILE_40B = 'data/Conservation/traitgym_complex_phastcon_llr_scores_40b.csv'

# ============================================================================
# STEP 0: DATA PREPARATION
# ============================================================================

def load_and_prepare_data(filepath: str, model_name: str) -> pd.DataFrame:
    """
    Load data and convert to long format with one row per variant-trait combination.
    
    Args:
        filepath: Path to CSV file
        model_name: Name of the model (e.g., '7B', '40B')
    
    Returns:
        DataFrame in long format with computed features
    """
    print(f"\n{'='*70}")
    print(f"Loading data for {model_name} model...")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(filepath, index_col=0)
    print(f"Original data shape: {df.shape}")
    
    # Explode traits (convert comma-separated traits to separate rows)
    df['trait'] = df['trait'].str.split(',')
    df_long = df.explode('trait')
    df_long['trait'] = df_long['trait'].str.strip()
    df_long = df_long.reset_index(drop=True)
    
    print(f"Long format shape (after exploding traits): {df_long.shape}")
    
    # Remove missing values
    df_long = df_long.dropna(subset=['phastcon', 'llr'])
    print(f"After removing missing values: {df_long.shape}")
    
    # Add small epsilon to avoid log(0) and log(1) issues
    epsilon = 1e-6
    df_long['phastcon_clipped'] = df_long['phastcon'].clip(epsilon, 1 - epsilon)
    
    # Compute logit(PhastCons)
    df_long['phastcons_logit'] = logit(df_long['phastcon_clipped'])
    
    # Compute |LLR|
    df_long['llr_abs'] = df_long['llr'].abs()
    
    # Add model identifier
    df_long['model'] = model_name
    
    # Create variant ID
    df_long['variant_id'] = (df_long['chrom'].astype(str) + ':' + 
                             df_long['pos'].astype(str) + ':' + 
                             df_long['ref'] + '>' + df_long['alt'])
    
    df_long['llr_abs_log1p'] = np.log1p(df_long['llr_abs'])

    return df_long

# ============================================================================
# STEP 1: SPEARMAN CORRELATION (PER GROUP)
# ============================================================================

def compute_spearman_per_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman correlation for each trait group.
    
    Args:
        df: Long-format DataFrame
    
    Returns:
        DataFrame with columns: trait_group, n, spearman_rho, spearman_p
    """
    print(f"\n{'='*70}")
    print("Computing Spearman correlations per trait group...")
    print(f"{'='*70}")
    
    results = []
    
    for trait in df['trait'].unique():
        trait_data = df[df['trait'] == trait]
        n = len(trait_data)
        
        if n >= MIN_GROUP_SIZE:
            # Compute Spearman correlation
            rho, p_value = stats.spearmanr(
                trait_data['phastcons_logit'], 
                trait_data['llr_abs']
            )
            
            results.append({
                'trait_group': trait,
                'n': n,
                'spearman_rho': rho,
                'spearman_p': p_value
            })
    
    results_df = pd.DataFrame(results)
    print(f"Analyzed {len(results_df)} trait groups (n >= {MIN_GROUP_SIZE})")
    print(f"Correlation range: [{results_df['spearman_rho'].min():.4f}, {results_df['spearman_rho'].max():.4f}]")
    
    return results_df

# ============================================================================
# STEP 2: PERMUTATION TESTING (PER GROUP)
# ============================================================================

def permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int = 10000) -> Tuple[float, float]:
    """
    Perform permutation test for Spearman correlation.
    
    Args:
        x: First variable (e.g., phastcons_logit)
        y: Second variable (e.g., llr_abs)
        n_perm: Number of permutations
    
    Returns:
        Tuple of (observed_rho, perm_p_value)
    """
    # Observed correlation
    rho_obs, _ = stats.spearmanr(x, y)
    
    # Permutation distribution
    rho_perm = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = np.random.permutation(y)
        rho_perm[i], _ = stats.spearmanr(x, y_perm)
    
    # Two-sided p-value
    p_value = (1 + np.sum(np.abs(rho_perm) >= np.abs(rho_obs))) / (n_perm + 1)
    
    return rho_obs, p_value

def compute_permutation_pvalues(df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute permutation p-values for all trait groups.
    
    Args:
        df: Long-format DataFrame
        results_df: DataFrame with Spearman results
    
    Returns:
        Updated results DataFrame with perm_p column
    """
    print(f"\n{'='*70}")
    print(f"Computing permutation p-values (n_perm={N_PERMUTATIONS})...")
    print(f"{'='*70}")
    
    perm_pvalues = []
    
    for idx, row in results_df.iterrows():
        trait = row['trait_group']
        trait_data = df[df['trait'] == trait]
        
        _, p_perm = permutation_test(
            trait_data['phastcons_logit'].values,
            trait_data['llr_abs'].values,
            n_perm=N_PERMUTATIONS
        )
        
        perm_pvalues.append(p_perm)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(results_df)} trait groups...")
    
    results_df['perm_p'] = perm_pvalues
    print(f"Permutation testing complete!")
    print(f"P-value range: [{min(perm_pvalues):.4f}, {max(perm_pvalues):.4f}]")
    
    return results_df

# ============================================================================
# STEP 3: FDR CORRECTION
# ============================================================================

def apply_fdr_correction(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Args:
        results_df: DataFrame with permutation p-values
    
    Returns:
        Updated results DataFrame with fdr_q column
    """
    print(f"\n{'='*70}")
    print("Applying Benjamini-Hochberg FDR correction...")
    print(f"{'='*70}")
    
    # Apply FDR correction
    reject, pvals_corrected, _, _ = multipletests(
        results_df['perm_p'], 
        alpha=FDR_ALPHA, 
        method='fdr_bh'
    )
    
    results_df['fdr_q'] = pvals_corrected
    results_df['significant'] = reject
    
    n_significant = reject.sum()
    print(f"Significant trait groups (FDR q < {FDR_ALPHA}): {n_significant}/{len(results_df)}")
    
    return results_df

# ============================================================================
# STEP 4: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for Spearman correlation.
    
    Args:
        x: First variable
        y: Second variable
        n_boot: Number of bootstrap iterations
        ci: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    rho_boot = np.zeros(n_boot)
    n = len(x)
    
    for i in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        rho_boot[i], _ = stats.spearmanr(x[idx], y[idx])
    
    alpha = 1 - ci
    lower = np.percentile(rho_boot, 100 * alpha / 2)
    upper = np.percentile(rho_boot, 100 * (1 - alpha / 2))
    
    return lower, upper

def compute_bootstrap_cis(df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for all trait groups.
    
    Args:
        df: Long-format DataFrame
        results_df: DataFrame with correlation results
    
    Returns:
        Updated results DataFrame with ci_lower and ci_upper columns
    """
    print(f"\n{'='*70}")
    print(f"Computing bootstrap 95% CIs (n_boot={N_BOOTSTRAP})...")
    print(f"{'='*70}")
    
    ci_lower = []
    ci_upper = []
    
    for idx, row in results_df.iterrows():
        trait = row['trait_group']
        trait_data = df[df['trait'] == trait]
        
        lower, upper = bootstrap_ci(
            trait_data['phastcons_logit'].values,
            trait_data['llr_abs'].values,
            n_boot=N_BOOTSTRAP
        )
        
        ci_lower.append(lower)
        ci_upper.append(upper)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(results_df)} trait groups...")
    
    results_df['ci_lower'] = ci_lower
    results_df['ci_upper'] = ci_upper
    print("Bootstrap CI computation complete!")
    
    return results_df

# ============================================================================
# FIGURE A: TRAIT-WISE CORRELATION SUMMARY
# ============================================================================

def plot_correlation_summary(results_df: pd.DataFrame, model_name: str, output_path: str):
    """
    Create main correlation summary figure with error bars and FDR significance.
    
    Args:
        results_df: DataFrame with all statistical results
        model_name: Name of the model
        output_path: Path to save the figure
    """
    print(f"\n{'='*70}")
    print(f"Creating Figure A: Correlation Summary ({model_name})...")
    print(f"{'='*70}")
    
    # Sort by correlation
    df_plot = results_df.sort_values('spearman_rho', ascending=True).reset_index(drop=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(df_plot) * 0.3)))
    
    # Color by significance
    colors = ['#d62728' if sig else '#1f77b4' for sig in df_plot['significant']]
    
    # Plot error bars
    for i, row in df_plot.iterrows():
        ax.errorbar(
            row['spearman_rho'], i,
            xerr=[[row['spearman_rho'] - row['ci_lower']], 
                  [row['ci_upper'] - row['spearman_rho']]],
            fmt='o',
            color=colors[i],
            markersize=8,
            capsize=3,
            capthick=1.5,
            linewidth=1.5,
            alpha=0.8
        )
    
    # Add correlation value annotations
    for i, row in df_plot.iterrows():
        # Position text: right side for positive, left side for negative
        if row['spearman_rho'] >= 0:
            x_pos = row['ci_upper'] + 0.02  # Right of upper CI
            ha = 'left'
        else:
            x_pos = row['ci_lower'] - 0.02  # Left of lower CI
            ha = 'right'
        
        ax.text(
            x_pos,
            i,
            f"{row['spearman_rho']:.3f}",
            va='center',
            ha=ha,
            fontsize=11,
            color=colors[i]
        )
    
    # Add vertical line at rho=0
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels and formatting
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['trait_group'], fontsize=12)
    ax.set_xlabel('Spearman ρ (PhastCons logit vs |LLR|)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Trait Group', fontsize=14, fontweight='bold')
    ax.set_title(f'Trait-wise Conservation-Alignment Correlation ({model_name})\n' +
                 f'with 95% Bootstrap CI and FDR Correction (α={FDR_ALPHA})',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Increase tick label size for x-axis
    ax.tick_params(axis='x', labelsize=12)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label=f'Significant (FDR q < {FDR_ALPHA})'),
        Patch(facecolor='#1f77b4', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()
# ============================================================================
# FIGURE B: REPRESENTATIVE SCATTER PLOTS
# ============================================================================

def plot_representative_scatters(df: pd.DataFrame, results_df: pd.DataFrame, 
                                 model_name: str, output_path: str, n_top: int = 3):
    """
    Create scatter plots for representative trait groups.
    
    Args:
        df: Long-format DataFrame
        results_df: DataFrame with statistical results
        model_name: Name of the model
        output_path: Path to save the figure
        n_top: Number of top/bottom groups to show
    """
    print(f"\n{'='*70}")
    print(f"Creating Figure B: Representative Scatter Plots ({model_name})...")
    print(f"{'='*70}")
    
    # Select representative groups
    df_sorted = results_df.sort_values('spearman_rho')
    
    # Bottom n (most negative)
    bottom_groups = df_sorted.head(n_top)['trait_group'].tolist()
    
    # Top n (most positive)
    top_groups = df_sorted.tail(n_top)['trait_group'].tolist()
    
    selected_groups = bottom_groups + top_groups
    
    # Create subplots
    n_plots = len(selected_groups)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, trait in enumerate(selected_groups):
        ax = axes[idx]
        
        # Get data for this trait
        trait_data = df[df['trait'] == trait]
        stats_row = results_df[results_df['trait_group'] == trait].iloc[0]
        
        # Scatter plot
        ax.scatter(
            trait_data['phastcons_logit'],
            # trait_data['llr_abs'],
            trait_data['llr_abs_log1p'],
            alpha=0.5,
            s=30,
            color='#1f77b4'
        )
        
         # Add LOWESS smoother
        if len(trait_data) >= 15:
            lowess_fit = lowess(
                trait_data['llr_abs_log1p'],
                trait_data['phastcons_logit'],
                frac=0.4,  # smooth, conservative
                return_sorted=True
            )
            ax.plot(
                lowess_fit[:, 0],
                lowess_fit[:, 1],
                color='red',
                linewidth=2,
                alpha=0.6
            )


        # Annotations
        ax.set_title(
            f"{trait}\n" +
            f"n={stats_row['n']}, ρ={stats_row['spearman_rho']:.3f}, " +
            f"q={stats_row['fdr_q']:.4f}",
            fontsize=10,
            fontweight='bold'
        )
        ax.set_xlabel('logit(PhastCons)', fontsize=9)
        # ax.set_ylabel('|LLR|', fontsize=9)
        ax.set_ylabel('log1p(|LLR|)', fontsize=9)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(
        f'Representative Scatter Plots: Conservation vs Evo2 |LLR| ({model_name})',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()

# ============================================================================
# FIGURE C: BINNED MONOTONICITY PLOT
# ============================================================================

def plot_binned_monotonicity(df: pd.DataFrame, results_df: pd.DataFrame,
                             model_name: str, output_path: str):
    """
    Create binned monotonicity plot showing mean |LLR| across PhastCons bins.
    
    Args:
        df: Long-format DataFrame
        results_df: DataFrame with statistical results
        model_name: Name of the model
        output_path: Path to save the figure
    """
    print(f"\n{'='*70}")
    print(f"Creating Figure C: Binned Monotonicity Plot ({model_name})...")
    print(f"{'='*70}")
    
    # Select groups for comparison
    df_sorted = results_df.sort_values('spearman_rho')
    
    # High correlation (top 5)
    high_corr_groups = df_sorted.tail(5)['trait_group'].tolist()
    
    # Low/negative correlation (bottom 5)
    low_corr_groups = df_sorted.head(5)['trait_group'].tolist()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, groups, label in zip(axes, 
                                  [high_corr_groups, low_corr_groups],
                                  ['High Positive Correlation', 'Low/Negative Correlation']):
        
        for trait in groups:
            trait_data = df[df['trait'] == trait].copy()
            stats_row = results_df[results_df['trait_group'] == trait].iloc[0]
            
            # Create bins based on PhastCons logit
            trait_data['bin'] = pd.qcut(
                trait_data['phastcons_logit'], 
                q=N_BINS, 
                labels=False, 
                duplicates='drop'
            )
            
            # Compute mean and SEM per bin
            bin_stats = trait_data.groupby('bin')['llr_abs'].agg(['mean', 'sem', 'count'])
            bin_centers = trait_data.groupby('bin')['phastcons_logit'].mean()
            
            # Plot
            ax.errorbar(
                bin_centers,
                bin_stats['mean'],
                yerr=bin_stats['sem'],
                marker='o',
                markersize=6,
                label=f"{trait} (ρ={stats_row['spearman_rho']:.3f})",
                capsize=3,
                alpha=0.8
            )
        
        ax.set_xlabel('PhastCons logit (binned)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean |LLR|', fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
    
    fig.suptitle(
        f'Binned Monotonicity: Mean |LLR| across Conservation Levels ({model_name})',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_model(filepath: str, model_name: str, output_dir: str = 'outputs'):
    """
    Run complete analysis pipeline for one model.
    
    Args:
        filepath: Path to input CSV file
        model_name: Name of the model (e.g., '7B', '40B')
        output_dir: Directory to save outputs
    """
    print(f"\n\n{'#'*70}")
    print(f"# ANALYZING {model_name} MODEL")
    print(f"{'#'*70}")

    os.makedirs(output_dir, exist_ok=True)
    # Load and prepare data
    df = load_and_prepare_data(filepath, model_name)
    
    # Step 1: Spearman correlation
    results_df = compute_spearman_per_group(df)
    
    # Step 2: Permutation testing
    results_df = compute_permutation_pvalues(df, results_df)
    
    # Step 3: FDR correction
    results_df = apply_fdr_correction(results_df)
    
    # Step 4: Bootstrap CIs
    results_df = compute_bootstrap_cis(df, results_df)
    
    # Sort by correlation for final output
    results_df = results_df.sort_values('spearman_rho', ascending=False)
    
    # Save results table
    output_table = f'{output_dir}/statistical_results_{model_name}.csv'
    results_df.to_csv(output_table, index=False)
    print(f"\nResults table saved to: {output_table}")
    
    # Generate figures
    plot_correlation_summary(
        results_df, model_name, 
        f'{output_dir}/FigureA_correlation_summary_{model_name}.png'
    )
    
    plot_representative_scatters(
        df, results_df, model_name,
        f'{output_dir}/FigureB_scatter_plots_{model_name}.png'
    )
    
    plot_binned_monotonicity(
        df, results_df, model_name,
        f'{output_dir}/FigureC_binned_monotonicity_{model_name}.png'
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY - {model_name} MODEL")
    print(f"{'='*70}")
    print(f"Total trait groups analyzed: {len(results_df)}")
    print(f"Significant groups (FDR q < {FDR_ALPHA}): {results_df['significant'].sum()}")
    print(f"\nTop 5 positive correlations:")
    print(results_df.head()[['trait_group', 'n', 'spearman_rho', 'fdr_q']])
    print(f"\nTop 5 negative correlations:")
    print(results_df.tail()[['trait_group', 'n', 'spearman_rho', 'fdr_q']])
    
    return results_df, df

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRAITGYM STATISTICAL ANALYSIS")
    print("Spearman Correlation + Permutation Testing + FDR Correction")
    print("="*70)
    
    # Analyze both models
    results_7b, df_7b = analyze_model(FILE_7B, '7B')
    results_40b, df_40b = analyze_model(FILE_40B, '40B')

    
    print("\n\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nAll outputs saved to: outputs")
    print("="*70 + "\n")