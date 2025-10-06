#!/usr/bin/env python3
"""
RLDDM Results Analysis Pipeline
Aggregates and analyzes SLURM results from fit_out/ directory structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from pathlib import Path
import glob
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_subject_summary(summary_file):
    """Load parameter estimates from a single subject's summary file"""
    try:
        df = pd.read_csv(summary_file, index_col=0)
        
        # Extract key parameters with diagnostics
        params = {}
        for param in ['alpha', 'a', 't0', 'scaler', 'log_scaler']:
            if param in df.index:
                params[f'{param}_mean'] = df.loc[param, 'mean']
                params[f'{param}_sd'] = df.loc[param, 'sd']
                if 'r_hat' in df.columns:
                    params[f'{param}_rhat'] = df.loc[param, 'r_hat']
                if 'ess_bulk' in df.columns:
                    params[f'{param}_ess'] = df.loc[param, 'ess_bulk']
        
        return params
    except Exception as e:
        print(f"Error loading {summary_file}: {e}")
        return None

def aggregate_by_condition():
    """Aggregate all results by condition (states)"""
    
    print("=== RLDDM Results Aggregation ===")
    
    all_results = []
    
    # Loop through each condition directory
    for states in [2, 3, 4, 5]:
        condition_dir = Path(f"fit_out/{states}state")
        
        if not condition_dir.exists():
            print(f"⚠️  Directory {condition_dir} not found, skipping")
            continue
        
        # Find all summary files in this condition
        summary_files = list(condition_dir.glob("subject*-summary.csv"))
        print(f"Condition {states}-state: Found {len(summary_files)} subjects")
        
        condition_results = []
        
        for summary_file in summary_files:
            # Extract subject ID from filename
            filename = summary_file.name
            subject_match = re.search(r'subject(\d+)', filename)
            if not subject_match:
                continue
            subject_id = int(subject_match.group(1))
            
            # Load parameter estimates
            params = load_subject_summary(summary_file)
            if params:
                result = {
                    'subject_id': subject_id,
                    'states': states,
                    'condition': f"{states}-state",
                    **params
                }
                condition_results.append(result)
                all_results.append(result)
        
        print(f"  Successfully loaded {len(condition_results)} subjects")
    
    if not all_results:
        print("❌ No results found in fit_out/ directories")
        return None
    
    # Create master DataFrame
    df = pd.DataFrame(all_results)
    df = df.sort_values(['states', 'subject_id']).reset_index(drop=True)
    
    print(f"\n✅ Total subjects loaded: {len(df)}")
    print("Summary by condition:")
    print(df.groupby('condition').size())
    
    return df

def convergence_diagnostics(df):
    """Check convergence diagnostics across all subjects"""
    
    print("\n=== Convergence Diagnostics ===")
    
    # R-hat diagnostics
    rhat_cols = [col for col in df.columns if col.endswith('_rhat')]
    if rhat_cols:
        print("\nR-hat summary (should be < 1.01):")
        for col in rhat_cols:
            param = col.replace('_rhat', '')
            rhat_vals = df[col].dropna()
            n_bad = (rhat_vals > 1.01).sum()
            print(f"  {param:10s}: mean={rhat_vals.mean():.3f}, max={rhat_vals.max():.3f}, n_bad={n_bad}")
    
    # ESS diagnostics
    ess_cols = [col for col in df.columns if col.endswith('_ess')]
    if ess_cols:
        print("\nEffective Sample Size summary (should be > 400):")
        for col in ess_cols:
            param = col.replace('_ess', '')
            ess_vals = df[col].dropna()
            n_bad = (ess_vals < 400).sum()
            print(f"  {param:10s}: mean={ess_vals.mean():.0f}, min={ess_vals.min():.0f}, n_bad={n_bad}")

def parameter_descriptives(df):
    """Descriptive statistics for each parameter by condition"""
    
    print("\n=== Parameter Descriptives by Condition ===")
    
    param_cols = [col for col in df.columns if col.endswith('_mean')]
    
    for param_col in param_cols:
        param = param_col.replace('_mean', '')
        print(f"\n{param.upper()} Parameter:")
        
        desc = df.groupby('condition')[param_col].agg(['count', 'mean', 'std', 'min', 'max'])
        print(desc.round(3))

def get_significance_stars(p_value):
    """Convert p-value to significance stars"""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""

def condition_comparisons(df):
    """Statistical comparisons between conditions"""
    
    print("\n=== Between-Condition Comparisons ===")
    
    param_cols = [col for col in df.columns if col.endswith('_mean')]
    conditions = sorted(df['condition'].unique())
    
    for param_col in param_cols:
        param = param_col.replace('_mean', '')
        print(f"\n{param.upper()} - One-way ANOVA across conditions:")
        
        # Prepare data for ANOVA
        groups = [df[df['condition'] == cond][param_col].dropna() for cond in conditions]
        
        try:
            f_stat, p_val = stats.f_oneway(*groups)
            anova_stars = get_significance_stars(p_val)
            print(f"  F({len(groups)-1}, {sum(len(g) for g in groups)-len(groups)}) = {f_stat:.3f}, p = {p_val:.4f} {anova_stars}")
            
            if p_val < 0.05:
                print("  🔍 Significant difference - performing pairwise comparisons:")
                for i, cond1 in enumerate(conditions):
                    for cond2 in conditions[i+1:]:
                        group1 = df[df['condition'] == cond1][param_col].dropna()
                        group2 = df[df['condition'] == cond2][param_col].dropna()
                        t_stat, t_p = stats.ttest_ind(group1, group2)
                        pairwise_stars = get_significance_stars(t_p)
                        print(f"    {cond1} vs {cond2}: t = {t_stat:.3f}, p = {t_p:.4f} {pairwise_stars}")
        except:
            print("  ❌ ANOVA failed (insufficient data)")

def create_parameter_plots(df):
    """Create comprehensive parameter visualization with significance stars"""
    
    print("\n=== Creating Parameter Plots ===")
    
    param_cols = [col for col in df.columns if col.endswith('_mean')]
    param_names = [col.replace('_mean', '') for col in param_cols]
    conditions = sorted(df['condition'].unique())
    
    # Set up the plot
    n_params = len(param_cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (param_col, param_name) in enumerate(zip(param_cols, param_names)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Box plot by condition
        df_clean = df[[param_col, 'condition']].dropna()
        if len(df_clean) > 0:
            sns.boxplot(data=df_clean, x='condition', y=param_col, ax=ax)
            sns.stripplot(data=df_clean, x='condition', y=param_col, ax=ax, 
                         color='black', alpha=0.5, size=3)
            
            # Perform ANOVA for this parameter
            groups = [df[df['condition'] == cond][param_col].dropna() for cond in conditions]
            
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                anova_stars = get_significance_stars(p_val)
                
                # Add significance stars to the title
                title_text = f'{param_name.upper()} by Condition'
                if anova_stars:
                    title_text += f' {anova_stars}'
                ax.set_title(title_text)
                
                # Add p-value annotation
                y_max = df_clean[param_col].max()
                y_range = df_clean[param_col].max() - df_clean[param_col].min()
                ax.text(0.02, 0.98, f'ANOVA p={p_val:.3f} {anova_stars}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add pairwise significance bars if ANOVA is significant
                if p_val < 0.05 and len(conditions) > 1:
                    y_pos = y_max + 0.1 * y_range
                    bar_height = 0.02 * y_range
                    
                    # Perform pairwise comparisons for significant pairs
                    significant_pairs = []
                    for j, cond1 in enumerate(conditions):
                        for k, cond2 in enumerate(conditions[j+1:], j+1):
                            group1 = df[df['condition'] == cond1][param_col].dropna()
                            group2 = df[df['condition'] == cond2][param_col].dropna()
                            if len(group1) > 0 and len(group2) > 0:
                                t_stat, t_p = stats.ttest_ind(group1, group2)
                                if t_p < 0.05:
                                    stars = get_significance_stars(t_p)
                                    significant_pairs.append((j, k, stars))
                    
                    # Draw significance bars for the most significant pairs (max 2 to avoid clutter)
                    for idx, (j, k, stars) in enumerate(significant_pairs[:2]):
                        y_bar = y_pos + idx * (bar_height * 3)
                        ax.plot([j, k], [y_bar, y_bar], 'k-', linewidth=1)
                        ax.plot([j, j], [y_bar - bar_height/2, y_bar + bar_height/2], 'k-', linewidth=1)
                        ax.plot([k, k], [y_bar - bar_height/2, y_bar + bar_height/2], 'k-', linewidth=1)
                        ax.text((j + k) / 2, y_bar + bar_height, stars, ha='center', va='bottom', fontweight='bold')
                
            except:
                ax.set_title(f'{param_name.upper()} by Condition')
            
            ax.set_xlabel('Condition')
            ax.set_ylabel(f'{param_name} estimate')
            ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('rlddm_parameter_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Parameter plots saved: rlddm_parameter_analysis.png")
    plt.show()

def create_correlation_analysis(df):
    """Analyze parameter correlations within each condition with significance stars"""
    
    print("\n=== Parameter Correlation Analysis ===")
    
    param_cols = [col for col in df.columns if col.endswith('_mean')]
    conditions = sorted(df['condition'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, condition in enumerate(conditions):
        if i >= len(axes):
            break
            
        ax = axes[i]
        cond_data = df[df['condition'] == condition][param_cols].dropna()
        
        if len(cond_data) > 5:  # Need minimum subjects for correlation
            # Compute correlation matrix and p-values
            corr_matrix = cond_data.corr()
            n_params = len(param_cols)
            p_matrix = np.ones((n_params, n_params))
            
            # Calculate p-values for correlations
            for j in range(n_params):
                for k in range(n_params):
                    if j != k:
                        r, p = stats.pearsonr(cond_data.iloc[:, j], cond_data.iloc[:, k])
                        p_matrix[j, k] = p
            
            # Create significance annotation matrix
            sig_matrix = np.full(corr_matrix.shape, '', dtype=object)
            for j in range(n_params):
                for k in range(n_params):
                    if j != k:
                        stars = get_significance_stars(p_matrix[j, k])
                        # Combine correlation value with stars
                        if not np.isnan(corr_matrix.iloc[j, k]):
                            corr_val = corr_matrix.iloc[j, k]
                            sig_matrix[j, k] = f'{corr_val:.2f}{stars}'
                    else:
                        sig_matrix[j, k] = f'{corr_matrix.iloc[j, k]:.2f}'
            
            # Plot heatmap with significance stars
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=sig_matrix, fmt='', 
                       cmap='coolwarm', center=0, ax=ax, cbar_kws={'shrink': 0.8},
                       annot_kws={'size': 8})
            ax.set_title(f'{condition} Parameter Correlations\n(n={len(cond_data)})')
            
            # Clean up labels
            labels = [col.replace('_mean', '') for col in param_cols]
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticklabels(labels, rotation=0)
            
            # Print significant correlations for this condition
            print(f"\n{condition} Significant Correlations:")
            sig_found = False
            for j in range(n_params):
                for k in range(j+1, n_params):  # Only upper triangle
                    if p_matrix[j, k] < 0.05:
                        r_val = corr_matrix.iloc[j, k]
                        p_val = p_matrix[j, k]
                        stars = get_significance_stars(p_val)
                        param1 = labels[j].upper()
                        param2 = labels[k].upper()
                        print(f"  {param1} ↔ {param2}: r = {r_val:.3f}, p = {p_val:.4f} {stars}")
                        sig_found = True
            if not sig_found:
                print("  No significant correlations found")
    
    plt.tight_layout()
    plt.savefig('rlddm_correlations.png', dpi=300, bbox_inches='tight')
    print("📊 Correlation plots saved: rlddm_correlations.png")
    plt.show()

def generate_summary_report(df):
    """Generate a comprehensive summary report"""
    
    print("\n=== Generating Summary Report ===")
    
    # Save detailed results
    df.to_csv('rlddm_aggregated_results.csv', index=False)
    print("💾 Detailed results saved: rlddm_aggregated_results.csv")
    
    # Create summary statistics table
    param_cols = [col for col in df.columns if col.endswith('_mean')]
    
    summary_stats = []
    for condition in sorted(df['condition'].unique()):
        cond_data = df[df['condition'] == condition]
        for param_col in param_cols:
            param = param_col.replace('_mean', '')
            values = cond_data[param_col].dropna()
            
            summary_stats.append({
                'condition': condition,
                'parameter': param,
                'n': len(values),
                'mean': values.mean(),
                'std': values.std(),
                'median': values.median(),
                'q25': values.quantile(0.25),
                'q75': values.quantile(0.75)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('rlddm_summary_statistics.csv', index=False)
    print("📋 Summary statistics saved: rlddm_summary_statistics.csv")
    
    return summary_df

def main():
    """Main analysis pipeline"""
    
    print("🚀 Starting RLDDM Analysis Pipeline")
    print("=" * 50)
    
    # Step 1: Aggregate results
    df = aggregate_by_condition()
    if df is None:
        return
    
    # Step 2: Check convergence
    convergence_diagnostics(df)
    
    # Step 3: Descriptive statistics
    parameter_descriptives(df)
    
    # Step 4: Between-condition comparisons
    condition_comparisons(df)
    
    # Step 5: Create visualizations
    create_parameter_plots(df)
    create_correlation_analysis(df)
    
    # Step 6: Generate reports
    summary_df = generate_summary_report(df)
    
    print("\n" + "=" * 50)
    print("✅ Analysis Complete!")
    print("\nOutput files:")
    print("  📊 rlddm_parameter_analysis.png")
    print("  📊 rlddm_correlations.png") 
    print("  💾 rlddm_aggregated_results.csv")
    print("  📋 rlddm_summary_statistics.csv")
    
    return df, summary_df

if __name__ == "__main__":
    df, summary_df = main()
