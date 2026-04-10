"""
Evo2 Scoring with Essentiality Labels - Complete Transcriptome Version
Handles both protein-coding and lncRNA genes
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from Bio import SeqIO

def load_essentiality_labels(csv_path):
    """Load essentiality labels"""
    print(f"\n{'='*70}")
    print("STEP 1: Loading Essentiality Labels")
    print(f"{'='*70}")
    
    labels_df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(labels_df)} genes with labels")
    
    if 'Consensus_Tier' in labels_df.columns:
        print(f"\n  Tier distribution:")
        for tier in sorted(labels_df['Consensus_Tier'].unique()):
            count = (labels_df['Consensus_Tier'] == tier).sum()
            print(f"    Tier {int(tier)}: {count} genes")
    
    return labels_df

def extract_gene_symbol(fasta_header):
    """
    Extract gene symbol from GENCODE FASTA header
    Format: ENST_ID|ENSG_ID|...|transcript_name|gene_symbol|length|
    """
    parts = fasta_header.split('|')
    if len(parts) >= 6:
        gene_symbol = parts[5].strip()
        return gene_symbol if gene_symbol != '-' else None
    return None

def read_fasta_sequences(fasta_path, gene_list, max_length=8192):
    """
    Read sequences from FASTA, keeping only genes in gene_list
    Takes longest transcript per gene
    """
    print(f"\n{'='*70}")
    print("STEP 2: Reading Sequences from FASTA")
    print(f"{'='*70}")
    print(f"FASTA: {fasta_path}")
    print(f"Filtering for {len(gene_list)} genes with labels")
    
    gene_set = set(gene_list)
    sequences = {}
    total_transcripts = 0
    skipped = 0
    
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Scanning FASTA"):
        total_transcripts += 1
        gene_symbol = extract_gene_symbol(record.id)
        
        if gene_symbol and gene_symbol in gene_set:
            sequence = str(record.seq).upper()
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            
            if gene_symbol not in sequences or len(sequence) > len(sequences[gene_symbol]):
                sequences[gene_symbol] = sequence
        else:
            skipped += 1
    
    print(f"\n✓ Total transcripts in FASTA: {total_transcripts}")
    print(f"✓ Unique genes matched: {len(sequences)}")
    print(f"  Transcripts skipped (no labels): {skipped}")
    
    missing = len(gene_set) - len(sequences)
    if missing > 0:
        print(f"  ⚠️  Missing from FASTA: {missing} genes ({missing/len(gene_set)*100:.1f}%)")
        missing_genes = gene_set - set(sequences.keys())
        print(f"     Examples: {list(missing_genes)[:5]}")
    
    return sequences

def score_sequences_batch(model, sequences, batch_size=16):
    """Score sequences using Evo2"""
    all_scores = []
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Scoring batches"):
        batch = sequences[i:i + batch_size]
        
        try:
            scores = model.score_sequences(batch)
            all_scores.extend(scores)
        except Exception as e:
            print(f"  ⚠️  Error scoring batch {i//batch_size + 1}: {e}")
            all_scores.extend([float('nan')] * len(batch))
    
    return all_scores

def main():
    parser = argparse.ArgumentParser(
        description='Score gene sequences with essentiality labels using Evo2'
    )
    parser.add_argument('--fasta', type=str,
                       default='/users/PAS0272/anuragshandilya94/gencode.v49.transcripts.fa',
                       help='Input FASTA file (complete transcriptome)')
    parser.add_argument('--labels_csv', type=str,
                       default='/users/PAS0272/anuragshandilya94/lncRNA_essentiality_labels.csv',
                       help='CSV file with essentiality labels')
    parser.add_argument('--output', type=str,
                       default='evo2_predictions_complete_40B.csv',
                       help='Output CSV file')
    parser.add_argument('--model_name', type=str, default='evo2_40b',
                       choices=['evo2_7b', 'evo2_40b', 'savanna_evo2_7b', 'savanna_evo2_40b'],
                       help='Evo2 model name')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for scoring')
    parser.add_argument('--max_length', type=int, default=8192,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Evo2 Gene Scoring with Essentiality Labels")
    print("="*70)
    print(f"FASTA: {args.fasta}")
    print(f"Labels: {args.labels_csv}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*70)
    
    if not os.path.exists(args.fasta):
        print(f"\n✗ Error: FASTA file not found: {args.fasta}")
        print("\nDid you download the complete transcriptome?")
        print("  wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.transcripts.fa.gz")
        sys.exit(1)
    
    if not os.path.exists(args.labels_csv):
        print(f"\n✗ Error: Labels file not found: {args.labels_csv}")
        sys.exit(1)
    
    labels_df = load_essentiality_labels(args.labels_csv)
    
    sequences_dict = read_fasta_sequences(args.fasta, labels_df['Gene'].tolist(), args.max_length)
    
    if len(sequences_dict) == 0:
        print("\n✗ Error: No sequences found")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("STEP 3: Validating Sequences")
    print(f"{'='*70}\n")
    
    gene_ids = []
    sequences = []
    seq_lengths = []
    errors = []
    
    for gene_id, sequence in sequences_dict.items():
        invalid_chars = set(sequence) - set('ACGT')
        if invalid_chars:
            errors.append(f"{gene_id}: Invalid characters: {invalid_chars}")
            continue
        
        gene_ids.append(gene_id)
        sequences.append(sequence)
        seq_lengths.append(len(sequence))
    
    print(f"✓ {len(sequences)} sequences ready for scoring")
    if errors:
        print(f"  ⚠️  {len(errors)} sequences skipped (invalid characters)")
    
    print(f"\n{'='*70}")
    print("STEP 4: Loading Evo2 Model")
    print(f"{'='*70}\n")
    
    try:
        from evo2 import Evo2
        evo2_model = Evo2(args.model_name)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("STEP 5: Scoring Sequences")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    scores = score_sequences_batch(evo2_model, sequences, args.batch_size)
    duration = (datetime.now() - start_time).total_seconds()
    
    results = pd.DataFrame({
        "Gene": gene_ids,
        "Evo2_Score": scores,
        "Sequence_Length": seq_lengths
    })
    
    final_df = pd.merge(results, labels_df, on='Gene', how='left')
    final_df = final_df[~final_df['Evo2_Score'].isna()]
    final_df = final_df.sort_values('Evo2_Score', ascending=False)
    
    final_df.to_csv(args.output, index=False)
    
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")
    print(f"✓ Successfully scored: {len(final_df)} genes")
    print(f"  Time: {duration:.2f}s ({duration/60:.2f} min)")
    print(f"  Avg time per gene: {duration/len(final_df):.2f}s")
    print(f"  Output: {args.output}\n")
    
    print("Evo2 Score Statistics:")
    print(f"  Mean:   {final_df['Evo2_Score'].mean():.4f}")
    print(f"  Median: {final_df['Evo2_Score'].median():.4f}")
    print(f"  Std:    {final_df['Evo2_Score'].std():.4f}")
    print(f"  Min:    {final_df['Evo2_Score'].min():.4f}")
    print(f"  Max:    {final_df['Evo2_Score'].max():.4f}\n")
    
    if 'Consensus_Tier' in final_df.columns:
        print("Evo2 Scores by Essentiality Tier:")
        print(f"{'Tier':<6} {'Count':<8} {'Mean Score':<12} {'Std':<12}")
        print("-" * 40)
        for tier in sorted(final_df['Consensus_Tier'].unique()):
            tier_data = final_df[final_df['Consensus_Tier'] == tier]
            print(f"{int(tier):<6} {len(tier_data):<8} "
                  f"{tier_data['Evo2_Score'].mean():<12.4f} "
                  f"{tier_data['Evo2_Score'].std():<12.4f}")
        
        from scipy import stats
        tier_scores = [final_df[final_df['Consensus_Tier'] == t]['Evo2_Score'].values 
                      for t in sorted(final_df['Consensus_Tier'].unique())]
        if len(tier_scores) > 1:
            h_stat, p_value = stats.kruskal(*tier_scores)
            print(f"\nKruskal-Wallis Test: H={h_stat:.3f}, p={p_value:.4e}")
            
            corr, corr_p = stats.spearmanr(final_df['Consensus_Tier'], final_df['Evo2_Score'])
            print(f"Spearman Correlation: ρ={corr:.3f}, p={corr_p:.4e}")
    
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()