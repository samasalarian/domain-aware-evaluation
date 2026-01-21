#!/usr/bin/env python3
"""
Biological Validity of Evo2-Generated Genomes
Statistical analysis of ORF lengths and Pfam annotation completeness
MAJOR REVISIONS:
1. Genome-level analysis (primary) to avoid pseudoreplication
2. Separate FDR correction for each metric family
3. Robust filename parsing with regex
4. Permutation test for % no-hit instead of t-test
5. ECDF plots and genome-level scatter plots
6. Consistent color scheme throughout
"""
import os
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from Bio import SeqIO
import warnings
warnings.filterwarnings('ignore')
# ============================================================================
# Configuration
# ============================================================================
# Input paths
PROT_FILTERED = Path("data/Validity/proteins")
HMMER_OUT = Path("data/Validity/hmmer")
# Output path
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
# Analysis parameters
E_VALUE_THRESHOLD = 1e-5
N_PERMUTATIONS = 10000  # For permutation tests
# Color scheme (consistent across all plots)
COLORS = {
    'Natural': '#2ecc71',
    'Evo2-40b': '#e74c3c',
    'Evo2-7b': '#c0392b'
}
# Plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
print("=" * 80)
print("Biological Naturalness Analysis (REVISED)")
print("=" * 80)
print(f"\nE-value threshold: {E_VALUE_THRESHOLD}")
print(f"Permutations: {N_PERMUTATIONS}")
print(f"Output directory: {OUTPUT_DIR}")
print()
# ============================================================================
# Helper Functions
# ============================================================================
def parse_model_from_filename(filename):
    """
    Robustly parse model and replicate from filename using regex
    """
    stem = Path(filename).stem
    # Find model (40b or 7b)
    model_match = re.search(r'(40b|7b)', stem, re.IGNORECASE)
    model = model_match.group(1) if model_match else 'unknown'
    # Find replicate number
    rep_match = re.search(r'rep(\d+)', stem, re.IGNORECASE)
    replicate = rep_match.group(1) if rep_match else 'unknown'
    return model, replicate
def permutation_test_one_sample(sample_values, null_value, n_perm=10000):
    """
    Permutation test for comparing sample mean to a single null value
    Two-sided test
    """
    observed_diff = np.mean(sample_values) - null_value
    # Generate null distribution by resampling with sign flips
    null_diffs = []
    centered_values = sample_values - null_value
    for _ in range(n_perm):
        # Randomly flip signs
        signs = np.random.choice([-1, 1], size=len(centered_values))
        perm_mean = np.mean(centered_values * signs) + null_value
        null_diffs.append(perm_mean - null_value)
    null_diffs = np.array(null_diffs)
    # Two-sided p-value
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    return p_value, null_diffs
# ============================================================================
# Step 1: Extract ORF Lengths from FASTA Files
# ============================================================================
def get_orf_lengths(fasta_file):
    """Extract protein lengths from FASTA file"""
    lengths = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths.append(len(record.seq))
    return lengths
print("Step 1: Extracting ORF lengths...")
print("-" * 80)
orf_data = []
# Natural genome
nat_file = PROT_FILTERED / "natural.faa"
if nat_file.exists():
    nat_lengths = get_orf_lengths(nat_file)
    for length in nat_lengths:
        orf_data.append({
            'genome_type': 'Natural',
            'genome_name': 'natural',
            'model': 'Natural',
            'replicate': 'NA',
            'orf_length': length
        })
    print(f"  Natural genome: {len(nat_lengths)} ORFs")
else:
    print(f"  ERROR: Natural genome file not found: {nat_file}")
    sys.exit(1)
# Synthetic genomes
synthetic_files = sorted(PROT_FILTERED.glob("corrected_*.faa"))
for faa_file in synthetic_files:
    base = faa_file.stem
    # Robust parsing
    model, rep = parse_model_from_filename(faa_file)
    lengths = get_orf_lengths(faa_file)
    for length in lengths:
        orf_data.append({
            'genome_type': 'Synthetic',
            'genome_name': base,
            'model': f"Evo2-{model}",
            'replicate': rep,
            'orf_length': length
        })
    print(f"  {base}: {len(lengths)} ORFs (model={model}, rep={rep})")
orf_df = pd.DataFrame(orf_data)
print(f"\nTotal ORFs extracted: {len(orf_df):,}")
print()
# ============================================================================
# Step 2: Extract Pfam Annotation Results
# ============================================================================
def parse_hmmer_domtbl(domtbl_file, evalue_threshold):
    """
    Parse HMMER domtblout file and return set of protein IDs with hits
    Uses full sequence E-value (column 7, 0-indexed = field[6])
    """
    proteins_with_hit = set()
    with open(domtbl_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split()
            if len(fields) < 23:  # HMMER domtblout has 23 columns
                continue
            # Column layout (0-indexed):
            # 0: target name
            # 3: query name (protein ID)
            # 6: full sequence E-value
            # 12: domain i-Evalue
            protein_id = fields[3]
            full_seq_evalue = float(fields[6])
            if full_seq_evalue <= evalue_threshold:
                proteins_with_hit.add(protein_id)
    return proteins_with_hit
def count_proteins_in_fasta(fasta_file):
    """Count total proteins in FASTA file"""
    return sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
print("Step 2: Parsing Pfam annotation results...")
print("-" * 80)
pfam_data = []
# Natural genome
nat_domtbl = HMMER_OUT / "natural.domtbl"
if nat_domtbl.exists():
    nat_total = count_proteins_in_fasta(PROT_FILTERED / "natural.faa")
    nat_with_hit = len(parse_hmmer_domtbl(nat_domtbl, E_VALUE_THRESHOLD))
    nat_no_hit = nat_total - nat_with_hit
    nat_no_hit_pct = (nat_no_hit / nat_total) * 100
    pfam_data.append({
        'genome_type': 'Natural',
        'genome_name': 'natural',
        'model': 'Natural',
        'replicate': 'NA',
        'total_proteins': nat_total,
        'proteins_with_hit': nat_with_hit,
        'proteins_no_hit': nat_no_hit,
        'pct_no_hit': nat_no_hit_pct
    })
    print(f"  Natural: {nat_total} proteins, {nat_with_hit} with hit, "
          f"{nat_no_hit} no hit ({nat_no_hit_pct:.2f}%)")
else:
    print(f"  ERROR: Natural HMMER results not found: {nat_domtbl}")
    sys.exit(1)
# Synthetic genomes
for faa_file in synthetic_files:
    base = faa_file.stem
    domtbl = HMMER_OUT / f"{base}.domtbl"
    if not domtbl.exists():
        print(f"  WARNING: HMMER results not found for {base}")
        continue
    # Robust parsing
    model, rep = parse_model_from_filename(faa_file)
    total = count_proteins_in_fasta(faa_file)
    with_hit = len(parse_hmmer_domtbl(domtbl, E_VALUE_THRESHOLD))
    no_hit = total - with_hit
    no_hit_pct = (no_hit / total) * 100
    pfam_data.append({
        'genome_type': 'Synthetic',
        'genome_name': base,
        'model': f"Evo2-{model}",
        'replicate': rep,
        'total_proteins': total,
        'proteins_with_hit': with_hit,
        'proteins_no_hit': no_hit,
        'pct_no_hit': no_hit_pct
    })
    print(f"  {base}: {total} proteins, {with_hit} with hit, "
          f"{no_hit} no hit ({no_hit_pct:.2f}%)")
pfam_df = pd.DataFrame(pfam_data)
print(f"\nTotal genomes analyzed: {len(pfam_df)}")
print()
# ============================================================================
# Step 3: Genome-Level Summary Statistics (PRIMARY ANALYSIS)
# ============================================================================
print("Step 3: Computing GENOME-LEVEL summary statistics...")
print("-" * 80)
# Compute median ORF length per genome (to avoid pseudoreplication)
genome_level_stats = orf_df.groupby(['genome_type', 'genome_name', 'model', 'replicate'])['orf_length'].agg([
    ('median_orf_length', 'median'),
    ('mean_orf_length', 'mean'),
    ('std_orf_length', 'std'),
    ('q25_orf_length', lambda x: x.quantile(0.25)),
    ('q75_orf_length', lambda x: x.quantile(0.75)),
    ('iqr_orf_length', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    ('n_orfs', 'count')
]).reset_index()
# Merge with Pfam data
genome_level_stats = genome_level_stats.merge(
    pfam_df, 
    on=['genome_type', 'genome_name', 'model', 'replicate']
)
# Save genome-level statistics
genome_stats_file = OUTPUT_DIR / "genome_level_statistics.csv"
genome_level_stats.to_csv(genome_stats_file, index=False)
print(f"  Saved: {genome_stats_file}")
print("\nGenome-Level Summary:")
print(genome_level_stats[['genome_name', 'model', 'median_orf_length', 'pct_no_hit']])
print()
# ============================================================================
# Step 4: Statistical Testing - Median ORF Length (GENOME-LEVEL)
# ============================================================================
print("Step 4: Statistical testing (GENOME-LEVEL)...")
print("-" * 80)
# Get natural genome median ORF length
natural_median_orf = genome_level_stats[
    genome_level_stats['genome_type'] == 'Natural'
]['median_orf_length'].values[0]
# Get synthetic genome medians by model
test_results_orf_genome = []
for model in ['Evo2-40b', 'Evo2-7b']:
    model_data = genome_level_stats[
        (genome_level_stats['genome_type'] == 'Synthetic') & 
        (genome_level_stats['model'] == model)
    ]
    if len(model_data) == 0:
        continue
    synthetic_medians = model_data['median_orf_length'].values
    # Permutation test
    p_val, null_dist = permutation_test_one_sample(
        synthetic_medians, 
        natural_median_orf, 
        n_perm=N_PERMUTATIONS
    )
    # Effect size (Cohen's d)
    mean_diff = np.mean(synthetic_medians) - natural_median_orf
    # Use SD of synthetic replicates (since natural n=1)
    std_synth = np.std(synthetic_medians, ddof=1)
    cohens_d = mean_diff / std_synth if std_synth > 0 else np.nan
    test_results_orf_genome.append({
        'comparison': f"{model} vs Natural",
        'metric': 'Median ORF Length (genome-level)',
        'n_natural': 1,
        'n_synthetic': len(synthetic_medians),
        'natural_value': natural_median_orf,
        'synthetic_mean': np.mean(synthetic_medians),
        'synthetic_std': std_synth,
        'mean_difference': mean_diff,
        'p_value_permutation': p_val,
        'cohens_d': cohens_d,
        'test_type': 'permutation'
    })
    print(f"{model}:")
    print(f"  Natural median: {natural_median_orf:.1f} aa")
    print(f"  Synthetic mean ± SD: {np.mean(synthetic_medians):.1f} ± {std_synth:.1f} aa")
    print(f"  Difference: {mean_diff:.1f} aa")
    print(f"  P-value (permutation): {p_val:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print()
# ============================================================================
# Step 5: Statistical Testing - % No-Hit (GENOME-LEVEL)
# ============================================================================
natural_no_hit_pct = genome_level_stats[
    genome_level_stats['genome_type'] == 'Natural'
]['pct_no_hit'].values[0]
test_results_pfam_genome = []
for model in ['Evo2-40b', 'Evo2-7b']:
    model_data = genome_level_stats[
        (genome_level_stats['genome_type'] == 'Synthetic') & 
        (genome_level_stats['model'] == model)
    ]
    if len(model_data) == 0:
        continue
    synthetic_no_hit_pcts = model_data['pct_no_hit'].values
    # Permutation test
    p_val, null_dist = permutation_test_one_sample(
        synthetic_no_hit_pcts, 
        natural_no_hit_pct, 
        n_perm=N_PERMUTATIONS
    )
    # Effect size
    mean_diff = np.mean(synthetic_no_hit_pcts) - natural_no_hit_pct
    std_synth = np.std(synthetic_no_hit_pcts, ddof=1)
    cohens_d = mean_diff / std_synth if std_synth > 0 else np.nan
    test_results_pfam_genome.append({
        'comparison': f"{model} vs Natural",
        'metric': '% No Pfam Hit (genome-level)',
        'n_natural': 1,
        'n_synthetic': len(synthetic_no_hit_pcts),
        'natural_value': natural_no_hit_pct,
        'synthetic_mean': np.mean(synthetic_no_hit_pcts),
        'synthetic_std': std_synth,
        'mean_difference': mean_diff,
        'p_value_permutation': p_val,
        'cohens_d': cohens_d,
        'test_type': 'permutation'
    })
    print(f"{model}:")
    print(f"  Natural % no-hit: {natural_no_hit_pct:.2f}%")
    print(f"  Synthetic mean ± SD: {np.mean(synthetic_no_hit_pcts):.2f} ± {std_synth:.2f}%")
    print(f"  Difference: {mean_diff:.2f}%")
    print(f"  P-value (permutation): {p_val:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print()
# Combine test results
test_results_genome_df = pd.DataFrame(
    test_results_orf_genome + test_results_pfam_genome
)
# FDR correction SEPARATELY for each metric family
orf_mask = test_results_genome_df['metric'].str.contains('ORF')
pfam_mask = test_results_genome_df['metric'].str.contains('Pfam')
test_results_genome_df['p_value_fdr'] = np.nan
test_results_genome_df['significant_fdr_0.05'] = False
if orf_mask.sum() > 0:
    _, pvals_fdr_orf, _, _ = multipletests(
        test_results_genome_df.loc[orf_mask, 'p_value_permutation'],
        method='fdr_bh'
    )
    test_results_genome_df.loc[orf_mask, 'p_value_fdr'] = pvals_fdr_orf
    test_results_genome_df.loc[orf_mask, 'significant_fdr_0.05'] = pvals_fdr_orf < 0.05
if pfam_mask.sum() > 0:
    _, pvals_fdr_pfam, _, _ = multipletests(
        test_results_genome_df.loc[pfam_mask, 'p_value_permutation'],
        method='fdr_bh'
    )
    test_results_genome_df.loc[pfam_mask, 'p_value_fdr'] = pvals_fdr_pfam
    test_results_genome_df.loc[pfam_mask, 'significant_fdr_0.05'] = pvals_fdr_pfam < 0.05
# Save
genome_test_file = OUTPUT_DIR / "statistical_tests_genome_level.csv"
test_results_genome_df.to_csv(genome_test_file, index=False)
print(f"  Saved: {genome_test_file}")
print()
# ============================================================================
# Step 6: Supplementary ORF-Level Analysis (for completeness)
# ============================================================================
print("Step 6: Supplementary ORF-level analysis...")
print("-" * 80)
# Aggregate ORFs by model for Mann-Whitney U test
natural_orf_lengths = orf_df[orf_df['genome_type'] == 'Natural']['orf_length'].values
orf_level_tests = []
for model in ['Evo2-40b', 'Evo2-7b']:
    model_orfs = orf_df[orf_df['model'] == model]['orf_length'].values
    if len(model_orfs) == 0:
        continue
    # Mann-Whitney U test
    statistic, pvalue = mannwhitneyu(
        natural_orf_lengths, 
        model_orfs, 
        alternative='two-sided'
    )
    # Effect size (rank-biserial correlation)
    n1 = len(natural_orf_lengths)
    n2 = len(model_orfs)
    U = statistic
    rank_biserial = 1 - (2 * U) / (n1 * n2)
    # Cohen's d
    pooled_std = np.sqrt(
        ((n1-1)*np.std(natural_orf_lengths, ddof=1)**2 + 
         (n2-1)*np.std(model_orfs, ddof=1)**2) / (n1 + n2 - 2)
    )
    cohens_d = (np.mean(model_orfs) - np.mean(natural_orf_lengths)) / pooled_std
    orf_level_tests.append({
        'comparison': f"{model} vs Natural",
        'metric': 'ORF Length (ORF-level, pooled)',
        'n_natural': n1,
        'n_synthetic': n2,
        'mean_natural': np.mean(natural_orf_lengths),
        'mean_synthetic': np.mean(model_orfs),
        'mann_whitney_U': statistic,
        'p_value': pvalue,
        'rank_biserial_r': rank_biserial,
        'cohens_d': cohens_d,
        'note': 'Supplementary - assumes independence of ORFs'
    })
orf_level_df = pd.DataFrame(orf_level_tests)
# FDR correction
if len(orf_level_df) > 0:
    _, pvals_fdr, _, _ = multipletests(orf_level_df['p_value'], method='fdr_bh')
    orf_level_df['p_value_fdr'] = pvals_fdr
    orf_level_df['significant_fdr_0.05'] = pvals_fdr < 0.05
supp_test_file = OUTPUT_DIR / "statistical_tests_orf_level_supplementary.csv"
orf_level_df.to_csv(supp_test_file, index=False)
print(f"  Saved: {supp_test_file}")
print()
# ============================================================================
# Step 7: Visualizations
# ============================================================================
print("Step 7: Generating visualizations...")
print("-" * 80)
# Prepare data for plotting
plot_data_agg = orf_df.copy()
plot_data_agg['label'] = plot_data_agg['model']
order = ['Natural', 'Evo2-40b', 'Evo2-7b']
palette = [COLORS[m] for m in order]
# Get data for density/ECDF plots
nat_data = orf_df[orf_df['genome_type'] == 'Natural']['orf_length']
model_40b_data = orf_df[orf_df['model'] == 'Evo2-40b']['orf_length']
model_7b_data = orf_df[orf_df['model'] == 'Evo2-7b']['orf_length']
# Plot 1: Density overlay - ORF Length
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(nat_data, bins=50, alpha=0.3, color=COLORS['Natural'], label='Natural', density=True)
nat_data.plot.kde(ax=ax, color=COLORS['Natural'], linewidth=2.5)
if len(model_40b_data) > 0:
    ax.hist(model_40b_data, bins=50, alpha=0.2, color=COLORS['Evo2-40b'], density=True)
    model_40b_data.plot.kde(ax=ax, color=COLORS['Evo2-40b'], linewidth=2.5, 
                            label='Synthetic Genome (40B)', alpha=0.8)
if len(model_7b_data) > 0:
    ax.hist(model_7b_data, bins=50, alpha=0.2, color=COLORS['Evo2-7b'], density=True)
    model_7b_data.plot.kde(ax=ax, color=COLORS['Evo2-7b'], linewidth=2.5, 
                           label='Synthetic Genome (7B)', alpha=0.8)
ax.set_xlabel('ORF Length (amino acids)', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title('ORF Length Density: Natural vs Evo2-Generated Genomes', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=11)
ax.set_xlim(0, 1500)
plt.tight_layout()
density_file = OUTPUT_DIR / "orf_length_density.png"
plt.savefig(density_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {density_file}")
plt.close()
# Plot 2: ECDF - ORF Length
fig, ax = plt.subplots(figsize=(10, 6))
# Natural
nat_sorted = np.sort(nat_data)
nat_ecdf = np.arange(1, len(nat_sorted)+1) / len(nat_sorted)
ax.plot(nat_sorted, nat_ecdf, color=COLORS['Natural'], linewidth=2.5, label='Natural')
# 40B
if len(model_40b_data) > 0:
    data_40b_sorted = np.sort(model_40b_data)
    ecdf_40b = np.arange(1, len(data_40b_sorted)+1) / len(data_40b_sorted)
    ax.plot(data_40b_sorted, ecdf_40b, color=COLORS['Evo2-40b'], linewidth=2.5, 
            label='Synthetic Genome (40B)', alpha=0.8)
# 7B
if len(model_7b_data) > 0:
    data_7b_sorted = np.sort(model_7b_data)
    ecdf_7b = np.arange(1, len(data_7b_sorted)+1) / len(data_7b_sorted)
    ax.plot(data_7b_sorted, ecdf_7b, color=COLORS['Evo2-7b'], linewidth=2.5, 
            label='Synthetic Genome (7B)', alpha=0.8)
ax.set_xlabel('ORF Length (amino acids)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
ax.set_title('Empirical CDF of ORF Lengths', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1500)
plt.tight_layout()
ecdf_file = OUTPUT_DIR / "orf_length_ecdf.png"
plt.savefig(ecdf_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {ecdf_file}")
plt.close()
# Plot 3: Combined figure (3 panels)
fig = plt.figure(figsize=(18, 6))
gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
# Panel A: Violin
ax1 = fig.add_subplot(gs[0, 0])
sns.violinplot(data=plot_data_agg, x='label', y='orf_length', order=order, 
               palette=palette, ax=ax1)
ax1.set_xlabel('Genome', fontsize=11, fontweight='bold')
ax1.set_ylabel('ORF Length (amino acids)', fontsize=11, fontweight='bold')
ax1.set_title('A. ORF Length Distribution', fontsize=12, fontweight='bold', loc='left')
ax1.set_xticklabels(['Natural', 'Synthetic (40B)', 'Synthetic (7B)'], fontsize=9)
# Panel B: Scatter
ax2 = fig.add_subplot(gs[0, 1])
for idx, row in genome_level_stats.iterrows():
    if row['genome_type'] == 'Natural':
        ax2.scatter(row['median_orf_length'], row['pct_no_hit'], 
                   s=300, color=COLORS['Natural'], marker='*', 
                   edgecolor='black', linewidth=2, zorder=10)
    else:
        color = COLORS[row['model']]
        ax2.scatter(row['median_orf_length'], row['pct_no_hit'], 
                   s=120, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Median ORF Length (aa)', fontsize=11, fontweight='bold')
ax2.set_ylabel('% No Pfam Hit', fontsize=11, fontweight='bold')
ax2.set_title('B. Genome-Level Comparison', fontsize=12, fontweight='bold', loc='left')
ax2.grid(True, alpha=0.3)
# Panel C: Bar
ax3 = fig.add_subplot(gs[0, 2])
bar_data = genome_level_stats.groupby('model')['pct_no_hit'].agg(['mean', 'std']).reset_index()
bar_data = bar_data.sort_values('model')
nat_row = pd.DataFrame({
    'model': ['Natural'],
    'mean': [natural_no_hit_pct],
    'std': [0]
})
bar_data = pd.concat([nat_row, bar_data], ignore_index=True)
colors_bar = [COLORS[m] for m in bar_data['model']]
ax3.bar(bar_data['model'], bar_data['mean'], yerr=bar_data['std'], 
        color=colors_bar, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Model', fontsize=11, fontweight='bold')
ax3.set_ylabel('% No Pfam Hit', fontsize=11, fontweight='bold')
ax3.set_title('C. Annotation Completeness', fontsize=12, fontweight='bold', loc='left')
ax3.set_xticklabels(['Natural', 'Synthetic (40B)', 'Synthetic (7B)'], fontsize=9)
plt.tight_layout()
combined_file = OUTPUT_DIR / "combined_figure.png"
plt.savefig(combined_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {combined_file}")
plt.close()
# ============================================================================
# Step 8: Publication Tables
# ============================================================================
print("\nStep 8: Generating publication tables...")
print("-" * 80)
# Table 1: Genome-level summary
pub_summary = []
nat_row = genome_level_stats[genome_level_stats['genome_type'] == 'Natural'].iloc[0]
pub_summary.append({
    'Genome': 'Natural (M. genitalium)',
    'N (ORFs)': f"{int(nat_row['n_orfs']):,}",
    'Median ORF Length (aa)': f"{nat_row['median_orf_length']:.0f}",
    'Mean ORF Length ± SD': f"{nat_row['mean_orf_length']:.1f} ± {nat_row['std_orf_length']:.1f}",
    '% No Pfam Hit': f"{nat_row['pct_no_hit']:.2f}"
})
for model in ['Evo2-40b', 'Evo2-7b']:
    model_data = genome_level_stats[genome_level_stats['model'] == model]
    if len(model_data) == 0:
        continue
    pub_summary.append({
        'Genome': f"{model} (n={len(model_data)})",
        'N (ORFs)': f"{int(model_data['n_orfs'].mean()):,}",
        'Median ORF Length (aa)': f"{model_data['median_orf_length'].mean():.0f} ± {model_data['median_orf_length'].std():.0f}",
        'Mean ORF Length ± SD': f"{model_data['mean_orf_length'].mean():.1f} ± {model_data['std_orf_length'].mean():.1f}",
        '% No Pfam Hit': f"{model_data['pct_no_hit'].mean():.2f} ± {model_data['pct_no_hit'].std():.2f}"
    })
pub_table_df = pd.DataFrame(pub_summary)
pub_table_file = OUTPUT_DIR / "publication_table1_genome_summary.csv"
pub_table_df.to_csv(pub_table_file, index=False)
print(f"  Saved: {pub_table_file}")
# Table 2: Statistical test results
def interpret_cohens_d(d):
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
test_summary = []
for _, row in test_results_genome_df.iterrows():
    if pd.notna(row['cohens_d']):
        test_summary.append({
            'Comparison': row['comparison'],
            'Metric': row['metric'],
            'Mean Difference': f"{row['mean_difference']:.2f}",
            "Cohen's d": f"{row['cohens_d']:.3f}",
            'Effect Size': interpret_cohens_d(row['cohens_d']),
            'P-value (permutation)': f"{row['p_value_permutation']:.4f}",
            'P-value (FDR)': f"{row['p_value_fdr']:.4f}",
            'Significant (FDR < 0.05)': 'Yes' if row['significant_fdr_0.05'] else 'No'
        })
test_table_df = pd.DataFrame(test_summary)
test_table_file = OUTPUT_DIR / "publication_table2_statistical_tests.csv"
test_table_df.to_csv(test_table_file, index=False)
print(f"  Saved: {test_table_file}")
print("\nPublication Table 1:")
print(pub_table_df.to_string(index=False))
print("\nPublication Table 2:")
print(test_table_df.to_string(index=False))
print()
# ============================================================================
# Final Summary
# ============================================================================
print("=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\n" + "=" * 80)