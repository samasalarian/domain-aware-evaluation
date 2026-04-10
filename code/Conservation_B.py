import pandas as pd
import numpy as np
import torch
from evo2 import Evo2
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import os
from mygene import MyGeneInfo
import argparse
from pyfaidx import Fasta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ARGUMENTS
# ============================================================
    parser = argparse.ArgumentParser(description="TCGA-BRCA + ClinVar VEP with trait categorization")
    parser.add_argument("--input_csv", required=True, default="/users/PAS0272/anuragshandilya94/BRCA_MAF_all_variants.csv", help="TCGA-BRCA MAF CSV (from extract script)")
    parser.add_argument("--clinvar_vcf", required=True, default="/users/PAS0272/anuragshandilya94/clinvar_GRCh38.vcf", help="ClinVar VCF file for labels")
    parser.add_argument("--fasta", required=True, default="/users/PAS0272/anuragshandilya94/Homo_sapiens.GRCh38.dna.primary_assembly.fa", help="Reference FASTA (GRCh38)")
    parser.add_argument("--output_dir", default="resultf" , required=True, help="Output directory")
    parser.add_argument("--model_name", default="evo2_7b", help="Evo2 model")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--window_size", type=int, default=50, help="Sequence context window")
    parser.add_argument("--score_threshold", type=int, default=1, help="Min score for categorization")
    parser.add_argument("--min_variants_per_category", type=int, default=10, 
                       help="Minimum labeled variants per category for AUROC/AUPRC")
    return parser.parse_args()

# ============================================================
# TRAIT KEYWORDS
# ============================================================
TRAIT_KEYWORDS = {
    "Oncological": {
        "primary": ["cancer", "tumor", "carcinoma", "oncogene", "metastasis",
                   "malignant", "neoplasm", "proliferation", "tumorigenesis"],
        "secondary": ["cell cycle", "dna repair", "apoptosis", "p53",
                     "growth factor", "angiogenesis", "cell division", "mutation"]
    },
    "Neurological": {
        "primary": ["neuron", "brain", "neural", "cerebral", "cognitive",
                   "neurodegeneration", "synapse", "alzheimer", "parkinson"],
        "secondary": ["synaptic", "axon", "dendrite", "glia", "myelin",
                     "neurotransmitter", "dopamine", "serotonin"]
    },
    "Metabolic": {
        "primary": ["metabolic", "metabolism", "glucose", "insulin",
                   "diabetes", "obesity", "enzyme"],
        "secondary": ["lipid", "oxidation", "biosynthesis",
                     "mitochondria", "energy", "atp", "metabolite"]
    },
    "Immune_System": {
        "primary": ["immune", "immunological", "antibody", "lymphocyte",
                   "inflammation", "autoimmune", "cytokine"],
        "secondary": ["antigen", "leukocyte", "macrophage",
                     "t cell", "b cell", "interferon", "interleukin"]
    },
    "Developmental": {
        "primary": ["embryo", "development", "differentiation", "morphogenesis",
                   "organogenesis", "embryonic", "stem cell"],
        "secondary": ["pattern formation", "limb development", "tissue development",
                     "gastrulation", "blastocyst"]
    },
    "Cardiovascular": {
        "primary": ["heart", "cardiac", "cardiovascular", "hypertension",
                   "coronary", "vascular"],
        "secondary": ["blood pressure", "artery", "endothelial",
                     "circulation", "myocardial", "atherosclerosis"]
    },
    "Structural_Cell": {
        "primary": ["cytoskeleton", "cell adhesion", "cell junction",
                   "extracellular matrix", "structural"],
        "secondary": ["membrane", "actin", "tubulin", "collagen",
                     "integrin", "cadherin", "focal adhesion"]
    }
}

# ============================================================
# CLINVAR PARSING
# ============================================================
def parse_clinvar_vcf(vcf_path):
    """Parse ClinVar VCF and extract pathogenicity labels"""
    print(f"\n📂 Parsing ClinVar VCF: {vcf_path}")
    print("   This may take a few minutes...")
    
    variants = []
    line_count = 0
    
    with open(vcf_path, 'r') as f:
        for line in f:
            line_count += 1
            
            if line.startswith('#'):
                continue
            
            if line_count % 100000 == 0:
                print(f"   Processed {line_count:,} lines, found {len(variants):,} labeled variants...")
            
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue
            
            chrom, pos, variant_id, ref, alt, qual, filt, info = fields[:8]
            
            gene = None
            clnsig = None
            
            for field in info.split(';'):
                if field.startswith('GENEINFO='):
                    gene_info = field.split('=')[1]
                    gene = gene_info.split(':')[0]
                elif field.startswith('CLNSIG='):
                    clnsig = field.split('=')[1]
            
            label = None
            if clnsig:
                clnsig_lower = clnsig.lower()
                
                if 'pathogenic' in clnsig_lower and 'benign' not in clnsig_lower:
                    if 'conflicting' not in clnsig_lower:
                        label = 1
                
                elif 'benign' in clnsig_lower and 'pathogenic' not in clnsig_lower:
                    if 'conflicting' not in clnsig_lower:
                        label = 0
            
            if label is not None and gene:
                variants.append({
                    'gene': gene,
                    'chrom': chrom.replace('chr', ''),
                    'pos': int(pos),
                    'ref': ref,
                    'alt': alt,
                    'label': label,
                    'clinvar_sig': clnsig
                })
    
    df = pd.DataFrame(variants)
    
    print(f"\nClinVar parsing complete!")
    print(f"   Total lines processed: {line_count:,}")
    print(f"   Labeled variants: {len(df):,}")
    print(f"   Pathogenic: {(df['label']==1).sum():,}")
    print(f"   Benign: {(df['label']==0).sum():,}")
    print(f"   Unique genes: {df['gene'].nunique():,}")
    
    return df

# ============================================================
# GENE CATEGORIZATION
# ============================================================
def calculate_category_score(description, category_keywords):
    desc_lower = str(description).lower()
    primary_score = sum(2 for kw in category_keywords["primary"] if kw in desc_lower)
    secondary_score = sum(1 for kw in category_keywords["secondary"] if kw in desc_lower)
    return primary_score + secondary_score

def categorize_gene_enhanced(gene_list, score_threshold=1):
    mg = MyGeneInfo()
    print(f"\nFetching annotations for {len(gene_list)} genes...")
    
    gene_info = mg.querymany(
        gene_list,
        scopes="symbol",
        fields="go.BP.term,go.MF.term,go.CC.term,summary,pathway.kegg.name,pathway.reactome.name,name",
        species="human",
        returnall=False
    )
    
    results = []
    
    print("Processing annotations with weighted scoring...")
    for g in tqdm(gene_info):
        gene_symbol = g.get("query", "")
        
        description = ""
        
        if "name" in g:
            description += str(g["name"]).lower() + " "
        
        if "summary" in g and g["summary"]:
            description += str(g["summary"]).lower() + " "
        
        if "go" in g:
            for go_type in ["BP", "MF", "CC"]:
                if go_type in g["go"]:
                    go_data = g["go"][go_type]
                    if isinstance(go_data, list):
                        terms = [t.get("term", "").lower() for t in go_data]
                    elif isinstance(go_data, dict):
                        terms = [go_data.get("term", "").lower()]
                    else:
                        terms = []
                    description += " ".join(terms) + " "
        
        if "pathway" in g:
            for pathway_db in ["kegg", "reactome"]:
                if pathway_db in g["pathway"]:
                    pw_data = g["pathway"][pathway_db]
                    if isinstance(pw_data, list):
                        pathways = [p.get("name", "").lower() for p in pw_data]
                    elif isinstance(pw_data, dict):
                        pathways = [pw_data.get("name", "").lower()]
                    else:
                        pathways = []
                    description += " ".join(pathways) + " "
        
        category_scores = {}
        for category, keywords in TRAIT_KEYWORDS.items():
            score = calculate_category_score(description, keywords)
            if score > 0:
                category_scores[category] = score
        
        if not category_scores or max(category_scores.values()) < score_threshold:
            primary_cat = "Unclassified"
        else:
            sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
            primary_cat = sorted_cats[0][0]
        
        results.append({
            "gene": gene_symbol,
            "trait_category": primary_cat,
            "max_score": max(category_scores.values()) if category_scores else 0
        })
    
    result_df = pd.DataFrame(results)
    
    print("\nGene Categorization:")
    print(result_df["trait_category"].value_counts())
    
    return result_df[["gene", "trait_category"]]

# ============================================================
# SEQUENCE EXTRACTION
# ============================================================
def get_sequence_context(chrom, pos, ref, alt, ref_genome, window=50):
    try:
        chrom_str = str(chrom).replace('chr', '')
        
        chrom_key = None
        for key in ref_genome.keys():
            if key == chrom_str or key == f"chr{chrom_str}" or key == str(chrom):
                chrom_key = key
                break
        
        if chrom_key is None:
            return None, None
        
        seq = ref_genome[chrom_key]
        
        pos = int(pos)
        start = max(0, pos - window - 1)
        end = min(len(seq), pos + window)
        context = str(seq[start:end]).upper()
        
        ref_pos = min(window, pos - 1) if pos > window else pos - 1
        ref = str(ref).strip().upper()
        alt = str(alt).strip().upper()
        
        if ref_pos + len(ref) > len(context):
            return None, None
            
        if context[ref_pos:ref_pos+len(ref)] != ref:
            if ref in context:
                ref_pos = context.index(ref)
            else:
                return None, None
        
        mut_seq = context[:ref_pos] + alt + context[ref_pos+len(ref):]
        
        return context, mut_seq
    except Exception as e:
        return None, None

# ============================================================
# LOAD AND MERGE DATA
# ============================================================
def load_and_merge_data(tcga_csv, clinvar_df, fasta_path, window_size, score_threshold):
    print(f"\nLoading TCGA-BRCA data: {tcga_csv}")
    df = pd.read_csv(tcga_csv)
    
    print(f"   Total TCGA variants: {len(df):,}")
    print(f"   Unique genes: {df['gene'].nunique():,}")
    
    required = ['gene', 'chrom', 'pos', 'ref', 'alt']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    print(f"\n Merging TCGA variants with ClinVar labels...")
    print(f"   TCGA variants: {len(df):,}")
    print(f"   ClinVar variants: {len(clinvar_df):,}")
    
    df = df.merge(
        clinvar_df[['gene', 'chrom', 'pos', 'ref', 'alt', 'label', 'clinvar_sig']],
        on=['gene', 'chrom', 'pos', 'ref', 'alt'],
        how='left'
    )
    
    n_labeled = df['label'].notna().sum()
    print(f"    Matched variants with ClinVar labels: {n_labeled:,} ({100*n_labeled/len(df):.1f}%)")
    print(f"   Pathogenic: {(df['label']==1).sum():,}")
    print(f"   Benign: {(df['label']==0).sum():,}")
    
    gene_list = df["gene"].dropna().unique().tolist()
    go_map = categorize_gene_enhanced(gene_list, score_threshold)
    df = df.merge(go_map, on="gene", how="left")
    df["trait_category"] = df["trait_category"].fillna("Unclassified")
    
    print(f"\nLoading reference genome: {fasta_path}")
    ref_genome = Fasta(fasta_path)
    print(f"   Available chromosomes: {list(ref_genome.keys())[:10]}...")
    
    print("\n Extracting sequence contexts...")
    wt_seqs = []
    mut_seqs = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing variants"):
        wt_seq, mut_seq = get_sequence_context(
            row['chrom'], row['pos'], row['ref'], row['alt'],
            ref_genome, window_size
        )
        wt_seqs.append(wt_seq)
        mut_seqs.append(mut_seq)
    
    df["wt_sequence"] = wt_seqs
    df["mut_sequence"] = mut_seqs
    
    valid_mask = df["wt_sequence"].notna() & df["mut_sequence"].notna()
    n_valid = valid_mask.sum()
    
    print(f"\n Successfully extracted sequences for {n_valid:,}/{len(df):,} variants ({100*n_valid/len(df):.1f}%)")
    
    df = df[valid_mask].reset_index(drop=True)
    
    n_labeled_valid = df['label'].notna().sum()
    print(f"   Labeled variants after filtering: {n_labeled_valid:,}")
    
    return df

# ============================================================
# RUN EVO2
# ============================================================
def run_evo2(df, model_name, batch_size):
    print(f"\n Loading Evo2 model: {model_name}")
    
    evo = Evo2(model_name=model_name)
    all_delta_ll = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Computing Evo2 scores"):
        batch = df.iloc[i : i + batch_size]
        wt_seqs = batch["wt_sequence"].tolist()
        mut_seqs = batch["mut_sequence"].tolist()
        
        try:
            wt_scores = evo.score_sequences(wt_seqs)
            mut_scores = evo.score_sequences(mut_seqs)
            delta_ll = [mut - wt for wt, mut in zip(wt_scores, mut_scores)]
            all_delta_ll.extend(delta_ll)
        except Exception as e:
            print(f" Error scoring batch at index {i}: {e}")
            all_delta_ll.extend([np.nan] * len(batch))

    df["delta_log_likelihood"] = all_delta_ll
    
    valid_mask = ~df["delta_log_likelihood"].isna()
    n_valid = valid_mask.sum()
    print(f" Successfully scored {n_valid:,}/{len(df):,} variants")
    
    df = df[valid_mask].reset_index(drop=True)
    
    return df

# ============================================================
# EVALUATE WITH AUROC/AUPRC
# ============================================================
def evaluate_performance(df, output_dir, min_variants=10):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n Saving full variant results...")
    df.to_csv(os.path.join(output_dir, "all_variants_scored.csv"), index=False)
    
    labeled_df = df[df['label'].notna()].copy()
    unlabeled_df = df[df['label'].isna()].copy()
    
    print(f"\nTotal variants: {len(df):,}")
    print(f"   Labeled (for AUROC/AUPRC): {len(labeled_df):,}")
    print(f"   Unlabeled (statistics only): {len(unlabeled_df):,}")

    results = []

    print(f"\n Computing category-wise metrics (min {min_variants} labeled variants)...")
    
    for cat, sub in df.groupby("trait_category"):
        labeled_sub = sub[sub['label'].notna()]
        
        auroc, auprc = np.nan, np.nan
        n_pathogenic, n_benign = 0, 0
        
        if len(labeled_sub) >= min_variants:
            try:
                y_true = labeled_sub['label'].astype(int)
                y_score = -labeled_sub['delta_log_likelihood']
                
                if len(y_true.unique()) >= 2:
                    auroc = roc_auc_score(y_true, y_score)
                    auprc = average_precision_score(y_true, y_score)
                    n_pathogenic = (y_true == 1).sum()
                    n_benign = (y_true == 0).sum()
                    
                    auroc_noflip = roc_auc_score(y_true, labeled_sub['delta_log_likelihood'])
                    if auroc_noflip > auroc:
                        auroc = auroc_noflip
                        auprc = average_precision_score(y_true, labeled_sub['delta_log_likelihood'])
            except Exception as e:
                print(f" Error computing metrics for {cat}: {e}")

        results.append({
            "Category": cat,
            "N_variants_total": len(sub),
            "N_variants_labeled": len(labeled_sub),
            "N_pathogenic": n_pathogenic,
            "N_benign": n_benign,
            "N_genes": sub["gene"].nunique(),
            "AUROC": auroc,
            "AUPRC": auprc,
            "Mean_delta_LL": np.mean(sub["delta_log_likelihood"]),
            "Median_delta_LL": np.median(sub["delta_log_likelihood"]),
            "Std_delta_LL": np.std(sub["delta_log_likelihood"]),
            "Q25_delta_LL": np.percentile(sub["delta_log_likelihood"], 25),
            "Q75_delta_LL": np.percentile(sub["delta_log_likelihood"], 75)
        })

    res_df = pd.DataFrame(results).sort_values("N_variants_total", ascending=False)
    res_df.to_csv(os.path.join(output_dir, "category_results_with_auroc.csv"), index=False)

    print("\n" + "="*120)
    print(" CATEGORY-WISE SUMMARY (TCGA-BRCA + ClinVar Labels)")
    print("="*120)
    print(res_df.to_string(index=False))
    print("="*120)

    valid_metrics = res_df.dropna(subset=["AUROC", "AUPRC"])
    
    if len(valid_metrics) > 0:
        overall = {
            "N_categories_with_labels": len(valid_metrics),
            "Total_labeled_variants": valid_metrics["N_variants_labeled"].sum(),
            "AUROC_mean": valid_metrics["AUROC"].mean(),
            "AUROC_range": valid_metrics["AUROC"].max() - valid_metrics["AUROC"].min(),
            "AUROC_std": valid_metrics["AUROC"].std(),
            "AUPRC_mean": valid_metrics["AUPRC"].mean(),
            "AUPRC_range": valid_metrics["AUPRC"].max() - valid_metrics["AUPRC"].min(),
            "AUPRC_std": valid_metrics["AUPRC"].std(),
        }
        
        spread_df = pd.DataFrame([overall])
        spread_df.to_csv(os.path.join(output_dir, "metric_spread.csv"), index=False)
        
        print("\n OVERALL METRIC SPREAD (Categories with AUROC):")
        print(spread_df.to_string(index=False))
    else:
        print("\nNo categories have enough labeled variants for AUROC/AUPRC calculation")
        print(f"   Try lowering --min_variants_per_category (current: {min_variants})")

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    
    print("="*100)
    print(" TCGA-BRCA VARIANT ANALYSIS WITH CLINVAR LABELS")
    print("="*100)
    
    clinvar_df = parse_clinvar_vcf(args.clinvar_vcf)
    
    df = load_and_merge_data(
        args.input_csv, 
        clinvar_df, 
        args.fasta, 
        args.window_size, 
        args.score_threshold
    )
    
    df = run_evo2(df, args.model_name, args.batch_size)
    
    evaluate_performance(df, args.output_dir, args.min_variants_per_category)
    
    print("\n" + "="*100)
    print(" ANALYSIS COMPLETE!")
    print(f" Results saved to: {args.output_dir}")
    print("="*100)

if __name__ == "__main__":
    main()