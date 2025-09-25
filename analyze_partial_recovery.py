#!/usr/bin/env python3
# analyze_partial_recovery.py
# Analyze whatever parameter recovery results we can extract

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob

def extract_from_cmdstan_csv(csv_file):
    """Extract parameter estimates from a cmdstan CSV file"""
    try:
        # Find where the data starts
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            
        header_line = None
        for i, line in enumerate(lines):
            if line.startswith('lp__'):
                header_line = i
                break
        
        if header_line is None:
            return None
            
        # Read the data
        df = pd.read_csv(csv_file, skiprows=header_line)
        
        # Remove any comment lines
        df = df[~df['lp__'].astype(str).str.startswith('#')]
        df = df.reset_index(drop=True)
        
        if len(df) == 0:
            return None
            
        # Extract parameters - use untransformed scaler
        params = ['alpha', 'a', 't0', 'scaler']
        if not all(param in df.columns for param in params):
            return None
            
        param_data = df[params].astype(float)
        
        # Compute estimates (using mean of samples)
        estimates = {
            'alpha_hat': param_data['alpha'].mean(),
            'a_hat': param_data['a'].mean(), 
            't0_hat': param_data['t0'].mean(),
            'scaler_hat': param_data['scaler'].mean(),  # Use untransformed scaler
            'n_samples': len(df),
            'source_file': csv_file
        }
        
        return estimates
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None

def find_parameter_recovery_files():
    """Find all potential parameter recovery result files"""
    
    # Look for cmdstan CSV files from parameter recovery attempts
    cmdstan_files = []
    
    # Check cmdstan_tmp for subject-specific directories
    cmdstan_pattern = "cmdstan_tmp/subject_*/rlddm_single_subject-*.csv"
    cmdstan_files.extend(glob.glob(cmdstan_pattern))
    
    # Also check for any other CSV files that might contain samples
    other_patterns = [
        "recovery_out/**/rlddm_single_subject-*.csv",
        "stan_out/rlddm_single_subject-*.csv"
    ]
    
    for pattern in other_patterns:
        files = glob.glob(pattern, recursive=True)
        cmdstan_files.extend(files)
    
    return list(set(cmdstan_files))

def extract_subject_id_from_path(file_path):
    """Try to extract subject ID from file path"""
    
    # Look for pattern like subject_N in path
    import re
    
    match = re.search(r'subject_(\d+)', file_path)
    if match:
        return int(match.group(1))
    
    # If not found, return None
    return None

def load_truth_parameters():
    """Load true parameters if they exist"""
    
    truth_paths = [
        "recovery_out/truth_params.csv",
        "truth_params.csv"
    ]
    
    for path in truth_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    return None

def create_recovery_analysis():
    """Main analysis function"""
    
    print("=== Parameter Recovery Analysis ===")
    print("Analyzing partial/interrupted parameter recovery results")
    print()
    
    # Find all potential result files
    csv_files = find_parameter_recovery_files()
    print(f"Found {len(csv_files)} potential CSV files")
    
    # Extract parameters from each file
    results = []
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        
        estimates = extract_from_cmdstan_csv(csv_file)
        if estimates:
            # Try to determine subject ID
            subj_id = extract_subject_id_from_path(csv_file)
            estimates['participant_id'] = subj_id if subj_id else 'unknown'
            estimates['file_path'] = csv_file
            
            results.append(estimates)
            print(f"  ✅ Extracted {estimates['n_samples']} samples")
        else:
            print(f"  ❌ Failed to extract parameters")
    
    if not results:
        print("❌ No parameter recovery results found")
        return None
    
    print(f"\n✅ Successfully extracted results from {len(results)} files")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Try to load truth parameters
    truth_df = load_truth_parameters()
    
    if truth_df is not None:
        print(f"📊 Found truth parameters for {len(truth_df)} subjects")
        
        # Merge with truth parameters where possible
        df_merged = df_results.merge(
            truth_df[['participant_id', 'alpha_true', 'a_true', 't0_true', 'scaler_true']], 
            on='participant_id', 
            how='left'
        )
    else:
        print("⚠️  No truth parameters found - cannot compute recovery metrics")
        df_merged = df_results.copy()
        for param in ['alpha', 'a', 't0', 'scaler']:
            df_merged[f'{param}_true'] = np.nan
    
    # Save results
    os.makedirs('recovery_out', exist_ok=True)
    results_file = 'recovery_out/partial_recovery_analysis.csv'
    df_merged.to_csv(results_file, index=False)
    
    print(f"\n📁 Results saved to: {results_file}")
    
    # Display results
    print(f"\n=== Results Summary ===")
    print(df_merged[['participant_id', 'alpha_hat', 'a_hat', 't0_hat', 'scaler_hat', 'n_samples']])
    
    # Compute recovery metrics if we have truth values
    if truth_df is not None and not df_merged['alpha_true'].isna().all():
        print(f"\n=== Parameter Recovery Metrics ===")
        
        for param in ['alpha', 'a', 't0', 'scaler']:
            true_col = f'{param}_true'
            est_col = f'{param}_hat'
            
            valid_mask = ~(df_merged[true_col].isna() | df_merged[est_col].isna())
            n_valid = valid_mask.sum()
            
            if n_valid > 0:
                true_vals = df_merged.loc[valid_mask, true_col]
                est_vals = df_merged.loc[valid_mask, est_col]
                
                bias = est_vals.mean() - true_vals.mean()
                rmse = np.sqrt(((est_vals - true_vals) ** 2).mean())
                
                if n_valid > 1:
                    corr = np.corrcoef(true_vals, est_vals)[0, 1]
                else:
                    corr = np.nan
                
                # Individual comparisons
                print(f"\n{param.upper()} Recovery:")
                for i, (idx, row) in enumerate(df_merged[valid_mask].iterrows()):
                    subj_id = row['participant_id']
                    true_val = row[true_col]
                    est_val = row[est_col]
                    error = est_val - true_val
                    print(f"  Subject {subj_id}: true={true_val:.3f}, est={est_val:.3f}, error={error:+.3f}")
                
                print(f"  Summary: bias={bias:+.3f}, rmse={rmse:.3f}, r={corr:.3f}")
            else:
                print(f"{param}: No valid data for recovery assessment")
    
    # Create visualization if matplotlib is available
    try:
        if truth_df is not None and not df_merged['alpha_true'].isna().all():
            create_recovery_plots(df_merged)
    except ImportError:
        print("Matplotlib not available - skipping plots")
    
    return df_merged

def create_recovery_plots(df):
    """Create parameter recovery visualization"""
    
    valid_data = df.dropna(subset=['alpha_true', 'alpha_hat', 'a_true', 'a_hat', 
                                   't0_true', 't0_hat', 'scaler_true', 'scaler_hat'])
    
    if len(valid_data) == 0:
        print("No valid data for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Parameter Recovery Results - Untransformed Scaler', fontsize=16)
    
    params = ['alpha', 'a', 't0', 'scaler']
    param_labels = ['Learning Rate (α)', 'Boundary (a)', 'Non-decision Time (t0)', 'Scaler (untransformed)']
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i//2, i%2]
        
        true_col = f'{param}_true'
        est_col = f'{param}_hat'
        
        if true_col in valid_data.columns and est_col in valid_data.columns:
            true_vals = valid_data[true_col]
            est_vals = valid_data[est_col]
            
            # Scatter plot
            ax.scatter(true_vals, est_vals, alpha=0.7, s=60)
            
            # Identity line
            min_val = min(true_vals.min(), est_vals.min())
            max_val = max(true_vals.max(), est_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Recovery')
            
            # Correlation
            if len(true_vals) > 1:
                r = np.corrcoef(true_vals, est_vals)[0, 1]
                ax.set_title(f'{label}\n(r = {r:.3f}, n = {len(true_vals)})')
            else:
                ax.set_title(f'{label}\n(n = {len(true_vals)})')
            
            ax.set_xlabel('True Value')
            ax.set_ylabel('Estimated Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('recovery_out/parameter_recovery_untransformed_plots.png', dpi=300, bbox_inches='tight')
    print("📈 Recovery plots saved to: recovery_out/parameter_recovery_untransformed_plots.png")
    plt.show()

if __name__ == "__main__":
    create_recovery_analysis()
