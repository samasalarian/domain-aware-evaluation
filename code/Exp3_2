"""
Experiment 3: Performance Variance in lncRNA Essentiality Prediction
Evaluates impact of essentiality strength and cellular context on Evo2 predictions

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("="*80)
print("EXPERIMENT 3: Performance Variance in lncRNA Essentiality Prediction")
print("="*80)

df = pd.read_csv('/users/PAS0272/anuragshandilya94/evo2_predictions_complete.csv')
print(f"\n✓ Loaded {len(df)} genes with Evo2 scores")

required_cols = ['Gene', 'Evo2_Score', 'Consensus_Tier', 
                 'HAP1_Label', 'HEK293FT_Label', 'K562_Label', 
                 'MDA-MB-231_Label', 'THP1_Label']

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"\n✗ ERROR: Missing required columns: {missing_cols}")
    print("\nYour CSV must contain per-cell-line labels.")
    print("Please ensure your essentiality labels CSV has individual cell line columns.")
    exit(1)

print(f"✓ All required columns present")
print(f"\nTier distribution:")
for tier in sorted(df['Consensus_Tier'].unique()):
    count = (df['Consensus_Tier'] == tier).sum()
    print(f"  Tier {int(tier)}: {count} genes")

print("\n" + "="*80)
print("TABLE X.1: Disaggregated AUROC Performance by Essentiality Tier")
print("="*80)

tier_results = []

tier_thresholds = [
    (1, "Essential in ≥1 Cell Line", [1, 2, 3, 4, 5]),
    (2, "Essential in ≥2 Cell Lines", [2, 3, 4, 5]),
    (3, "Essential in ≥3 Cell Lines", [3, 4, 5]),
    (4, "Essential in ≥4 Cell Lines", [4, 5]),
    (5, "Essential in 5 Cell Lines", [5])
]

for threshold, description, positive_tiers in tier_thresholds:
    tier_df = df[df['Consensus_Tier'].isin(positive_tiers + [0])].copy()
    tier_df['Binary_Label'] = tier_df['Consensus_Tier'].isin(positive_tiers).astype(int)
    
    n_pos = tier_df['Binary_Label'].sum()
    n_neg = len(tier_df) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        print(f"  ⚠️  Skipping {description}: insufficient samples")
        continue
    
    auroc = roc_auc_score(tier_df['Binary_Label'], tier_df['Evo2_Score'])
    
    auprc = average_precision_score(tier_df['Binary_Label'], tier_df['Evo2_Score'])
    
    pos_scores = tier_df[tier_df['Binary_Label'] == 1]['Evo2_Score']
    neg_scores = tier_df[tier_df['Binary_Label'] == 0]['Evo2_Score']
    u_stat, p_val = stats.mannwhitneyu(pos_scores, neg_scores, alternative='greater')
    
    pooled_std = np.sqrt((pos_scores.std()**2 + neg_scores.std()**2) / 2)
    cohens_d = (pos_scores.mean() - neg_scores.mean()) / pooled_std
    
    tier_results.append({
        'Threshold': threshold,
        'Description': description,
        'N_Positive': n_pos,
        'N_Negative': n_neg,
        'AUROC': auroc,
        'AUPRC': auprc,
        'Cohens_d': cohens_d,
        'p_value': p_val,
        'Mean_Pos': pos_scores.mean(),
        'Mean_Neg': neg_scores.mean()
    })

tier_df_results = pd.DataFrame(tier_results)

print("\n" + "-"*80)
print(f"{'Essentiality Tier':<35} {'Evo2 AUROC':<15} {'N (Pos/Neg)':<20} {'p-value':<15}")
print("-"*80)
for _, row in tier_df_results.iterrows():
    print(f"{row['Description']:<35} {row['AUROC']:<15.3f} "
          f"{row['N_Positive']}/{row['N_Negative']:<15} {row['p_value']:<15.2e}")
print("-"*80)

print("\n" + "="*80)
print("TABLE X.2: Disaggregated AUROC Performance by Individual Cell Line")
print("="*80)

cell_lines = {
    'HAP1': 'Chronic Myeloid Leukemia (Near-haploid)',
    'HEK293FT': 'Embryonic Kidney (Aneuploid, engineered)',
    'K562': 'Chronic Myeloid Leukemia (Pseudo-triploid)',
    'MDA-MB-231': 'Breast Adenocarcinoma TNBC (Highly aneuploid)',
    'THP1': 'Acute Monocytic Leukemia (Suspension)'
}

cell_line_results = []

for cell_line, description in cell_lines.items():
    label_col = f'{cell_line}_Label'
    
    cl_df = df[[label_col, 'Evo2_Score']].dropna()
    
    n_pos = (cl_df[label_col] == 1).sum()
    n_neg = (cl_df[label_col] == 0).sum()
    
    if n_pos == 0 or n_neg == 0:
        print(f"  ⚠️  Skipping {cell_line}: insufficient samples")
        continue
    
    auroc = roc_auc_score(cl_df[label_col], cl_df['Evo2_Score'])
    
    auprc = average_precision_score(cl_df[label_col], cl_df['Evo2_Score'])
    
    pos_scores = cl_df[cl_df[label_col] == 1]['Evo2_Score']
    neg_scores = cl_df[cl_df[label_col] == 0]['Evo2_Score']
    u_stat, p_val = stats.mannwhitneyu(pos_scores, neg_scores, alternative='greater')
    
    pooled_std = np.sqrt((pos_scores.std()**2 + neg_scores.std()**2) / 2)
    cohens_d = (pos_scores.mean() - neg_scores.mean()) / pooled_std
    
    cell_line_results.append({
        'Cell_Line': cell_line,
        'Description': description,
        'N_Positive': n_pos,
        'N_Negative': n_neg,
        'AUROC': auroc,
        'AUPRC': auprc,
        'Cohens_d': cohens_d,
        'p_value': p_val,
        'Mean_Pos': pos_scores.mean(),
        'Mean_Neg': neg_scores.mean()
    })

cell_line_df_results = pd.DataFrame(cell_line_results)

print("\n" + "-"*80)
print(f"{'Cell Line':<15} {'Tissue/Features':<45} {'AUROC':<10} {'p-value':<15}")
print("-"*80)
for _, row in cell_line_df_results.iterrows():
    print(f"{row['Cell_Line']:<15} {row['Description']:<45} "
          f"{row['AUROC']:<10.3f} {row['p_value']:<15.2e}")
print("-"*80)

print("\n" + "="*80)
print("QUANTITATIVE VARIANCE ANALYSIS")
print("="*80)

tier_aurocs = tier_df_results['AUROC'].values
tier_range = tier_aurocs.max() - tier_aurocs.min()
tier_std = tier_aurocs.std()
tier_mean = tier_aurocs.mean()

cell_aurocs = cell_line_df_results['AUROC'].values
cell_range = cell_aurocs.max() - cell_aurocs.min()
cell_std = cell_aurocs.std()
cell_mean = cell_aurocs.mean()

print(f"\nPerformance Variance Across ESSENTIALITY TIERS:")
print(f"  Mean AUROC:          {tier_mean:.3f}")
print(f"  Range:               {tier_range:.3f} ({tier_aurocs.min():.3f} to {tier_aurocs.max():.3f})")
print(f"  Standard Deviation:  {tier_std:.3f}")
print(f"  Coefficient of Variation: {(tier_std/tier_mean)*100:.1f}%")

print(f"\nPerformance Variance Across CELL LINES:")
print(f"  Mean AUROC:          {cell_mean:.3f}")
print(f"  Range:               {cell_range:.3f} ({cell_aurocs.min():.3f} to {cell_aurocs.max():.3f})")
print(f"  Standard Deviation:  {cell_std:.3f}")
print(f"  Coefficient of Variation: {(cell_std/cell_mean)*100:.1f}%")

print(f"\nDOMINANT SOURCE OF VARIABILITY:")
if cell_range > tier_range:
    print(f"  ✓ CELLULAR CONTEXT dominates")
    print(f"    Range ratio (Cell/Tier): {cell_range/tier_range:.2f}x")
else:
    print(f"  ✓ ESSENTIALITY STRENGTH dominates")
    print(f"    Range ratio (Tier/Cell): {tier_range/cell_range:.2f}x")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
x_pos = range(len(tier_df_results))
bars = ax.bar(x_pos, tier_df_results['AUROC'], color='steelblue', alpha=0.8, edgecolor='black')

for i, (auroc, desc) in enumerate(zip(tier_df_results['AUROC'], 
                                       tier_df_results['Description'])):
    ax.text(i, auroc + 0.01, f'{auroc:.3f}', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels([f"≥{row['Threshold']}" for _, row in tier_df_results.iterrows()], 
                    fontsize=11)
ax.set_xlabel('Essentiality Consensus (Cell Lines)', fontsize=13, fontweight='bold')
ax.set_ylabel('AUROC', fontsize=13, fontweight='bold')
ax.set_title('A. Performance by Essentiality Tier\n(Signal Strength)', 
             fontsize=14, fontweight='bold')
ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Random')
ax.set_ylim([0.45, 1.0])
ax.grid(alpha=0.3, axis='y')
ax.legend(fontsize=10)

ax.annotate('', xy=(0, tier_aurocs.min()), xytext=(len(tier_df_results)-1, tier_aurocs.max()),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(len(tier_df_results)/2, (tier_aurocs.min() + tier_aurocs.max())/2, 
        f'Range: {tier_range:.3f}', fontsize=10, color='red', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax = axes[1]
x_pos = range(len(cell_line_df_results))
bars = ax.bar(x_pos, cell_line_df_results['AUROC'], 
              color='coral', alpha=0.8, edgecolor='black')

for i, auroc in enumerate(cell_line_df_results['AUROC']):
    ax.text(i, auroc + 0.01, f'{auroc:.3f}', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(cell_line_df_results['Cell_Line'], fontsize=11, rotation=0)
ax.set_xlabel('Cell Line', fontsize=13, fontweight='bold')
ax.set_ylabel('AUROC', fontsize=13, fontweight='bold')
ax.set_title('B. Performance by Cellular Context\n(Genetic Background)', 
             fontsize=14, fontweight='bold')
ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Random')
ax.set_ylim([0.45, 1.0])
ax.grid(alpha=0.3, axis='y')
ax.legend(fontsize=10)

ax.annotate('', xy=(0, cell_aurocs.min()), xytext=(len(cell_line_df_results)-1, cell_aurocs.max()),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(len(cell_line_df_results)/2, (cell_aurocs.min() + cell_aurocs.max())/2, 
        f'Range: {cell_range:.3f}', fontsize=10, color='red',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('experiment3_performance_variance.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: experiment3_performance_variance.png")

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Essentiality Tier\n(Signal Strength)', 'Cell Line\n(Cellular Context)']
means = [tier_mean, cell_mean]
stds = [tier_std, cell_std]
ranges = [tier_range, cell_range]

x_pos = [0, 1]
width = 0.35

bars1 = ax.bar([p - width/2 for p in x_pos], means, width, 
               yerr=stds, capsize=10, label='Mean ± SD', 
               color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')

bars2 = ax.bar([p + width/2 for p in x_pos], ranges, width,
               label='Range', color=['steelblue', 'coral'], 
               alpha=0.4, edgecolor='black', hatch='///')

ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylabel('AUROC Variance Metrics', fontsize=13, fontweight='bold')
ax.set_title('Performance Variance: Signal Strength vs Cellular Context', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

for i, (m, s, r) in enumerate(zip(means, stds, ranges)):
    ax.text(i - width/2, m + s + 0.02, f'μ={m:.3f}\nσ={s:.3f}', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(i + width/2, r + 0.02, f'{r:.3f}', 
            ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('experiment3_variance_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: experiment3_variance_comparison.png")

tier_df_results.to_csv('experiment3_tier_results.csv', index=False)
print(f"✓ Saved: experiment3_tier_results.csv")

cell_line_df_results.to_csv('experiment3_cellline_results.csv', index=False)
print(f"✓ Saved: experiment3_cellline_results.csv")

with open('experiment3_summary_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("EXPERIMENT 3: Performance Variance Analysis\n")
    f.write("Impact of Essentiality Strength and Cellular Context\n")
    f.write("="*80 + "\n\n")
    
    f.write("TABLE X.1: AUROC by Essentiality Tier\n")
    f.write("-"*80 + "\n")
    f.write(tier_df_results.to_string(index=False))
    f.write("\n\n")
    
    f.write("TABLE X.2: AUROC by Cell Line\n")
    f.write("-"*80 + "\n")
    f.write(cell_line_df_results.to_string(index=False))
    f.write("\n\n")
    
    f.write("VARIANCE ANALYSIS\n")
    f.write("-"*80 + "\n")
    f.write(f"Essentiality Tier Range: {tier_range:.3f}\n")
    f.write(f"Essentiality Tier StdDev: {tier_std:.3f}\n")
    f.write(f"Cell Line Range: {cell_range:.3f}\n")
    f.write(f"Cell Line StdDev: {cell_std:.3f}\n")
    f.write(f"\nDominant Variance Source: ")
    f.write("CELLULAR CONTEXT\n" if cell_range > tier_range else "ESSENTIALITY STRENGTH\n")

print(f"✓ Saved: experiment3_summary_report.txt")

print("\n" + "="*80)
print("EXPERIMENT 3 ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey Finding: {'Cellular context' if cell_range > tier_range else 'Essentiality strength'} "
      f"dominates prediction variance")
print(f"   Range ratio: {max(cell_range, tier_range) / min(cell_range, tier_range):.2f}x")