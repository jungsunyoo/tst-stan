#!/usr/bin/env python3
# collect_recovery_results.py
# Robust collection of parameter recovery results from various sources

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import argparse
import sys

def find_summary_files():
    """Find all potential summary files from parameter recovery"""
    
    # Look in multiple locations
    locations = [
        "stan_out/*-summary.csv",
        "recovery_out/**/recovery_result.csv", 
        "recovery_out/**/*-summary.csv",
        "cmdstan_tmp/**/rlddm_single_subject-*-summary.csv"
    ]
    
    all_files = []
    for pattern in locations:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    return list(set(all_files))  # Remove duplicates

def extract_from_summary_file(summary_file):
    """Extract parameters from a summary CSV file"""
    try:
        df = pd.read_csv(summary_file, index_col=0)
        
        # Check if this looks like our model's summary
        required_params = ['alpha', 'a', 't0', 'scaler']
        if not all(param in df.index for param in required_params):
            return None
            
        result = {
            'alpha_hat': float(df.loc['alpha', 'mean']),
            'a_hat': float(df.loc['a', 'mean']),
            't0_hat': float(df.loc['t0', 'mean']),
            'scaler_hat': float(df.loc['scaler', 'mean']),
            'source_file': summary_file,
            'file_mtime': os.path.getmtime(summary_file)
        }
        
        # Add log_scaler if available
        if 'log_scaler' in df.index:
            result['log_scaler_hat'] = float(df.loc['log_scaler', 'mean'])
        
        return result
        
        return result
        
    except Exception as e:
        print(f"Error processing {summary_file}: {e}")
        return None

def load_truth_parameters(recovery_dir="recovery_out"):
    """Load true parameters if available"""
    truth_file = Path(recovery_dir) / "truth_params.csv"
    if truth_file.exists():
        return pd.read_csv(truth_file)
    return None

def match_results_to_subjects(results, truth_df=None):
    """Try to match extracted results to subject IDs"""
    
    if truth_df is None:
        # Can't match without truth parameters
        for i, result in enumerate(results):
            result['participant_id'] = f'unknown_{i+1}'
            result['alpha_true'] = np.nan
            result['a_true'] = np.nan  
            result['t0_true'] = np.nan
            result['scaler_true'] = np.nan
            result['log_scaler_true'] = np.nan
        return results
    
    # For now, just assign sequential IDs - would need more sophisticated matching
    # based on file timestamps and expected subject order
    matched_results = []
    for i, result in enumerate(results):
        subj_id = i + 1
        if subj_id <= len(truth_df):
            truth_row = truth_df[truth_df['participant_id'] == subj_id].iloc[0]
            result['participant_id'] = subj_id
            result['alpha_true'] = truth_row['alpha_true']
            result['a_true'] = truth_row['a_true']
            result['t0_true'] = truth_row['t0_true'] 
            result['scaler_true'] = truth_row['scaler_true']
            # Add log_scaler_true if available
            if 'log_scaler_true' in truth_row:
                result['log_scaler_true'] = truth_row['log_scaler_true']
            else:
                result['log_scaler_true'] = np.nan
        else:
            result['participant_id'] = f'unknown_{subj_id}'
            result['alpha_true'] = np.nan
            result['a_true'] = np.nan
            result['t0_true'] = np.nan
            result['scaler_true'] = np.nan
            result['log_scaler_true'] = np.nan
        matched_results.append(result)
    
    return matched_results

def main():
    parser = argparse.ArgumentParser(description="Collect parameter recovery results from various sources")
    parser.add_argument("--outdir", default="recovery_out", help="Output directory")
    parser.add_argument("--min_time", help="Minimum file modification time (YYYY-MM-DD HH:MM)")
    args = parser.parse_args()
    
    print("=== Collecting Parameter Recovery Results ===")
    
    # Find all potential summary files
    summary_files = find_summary_files()
    print(f"Found {len(summary_files)} potential summary files")
    
    # Filter by time if specified
    if args.min_time:
        import datetime
        min_timestamp = datetime.datetime.strptime(args.min_time, "%Y-%m-%d %H:%M").timestamp()
        summary_files = [f for f in summary_files if os.path.getmtime(f) >= min_timestamp]
        print(f"After time filtering: {len(summary_files)} files")
    
    # Extract parameters from each file
    results = []
    for summary_file in summary_files:
        result = extract_from_summary_file(summary_file)
        if result:
            results.append(result)
            print(f"✅ Extracted from {summary_file}")
        else:
            print(f"❌ Failed to extract from {summary_file}")
    
    if not results:
        print("❌ No valid parameter recovery results found")
        return
    
    print(f"\nSuccessfully extracted parameters from {len(results)} files")
    
    # Sort by file modification time
    results.sort(key=lambda x: x['file_mtime'])
    
    # Try to load truth parameters
    truth_df = load_truth_parameters(args.outdir)
    if truth_df is not None:
        print(f"Loaded truth parameters for {len(truth_df)} subjects")
    else:
        print("No truth parameters found - results will have unknown true values")
    
    # Match results to subjects
    matched_results = match_results_to_subjects(results, truth_df)
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    df = pd.DataFrame(matched_results)
    
    # Reorder columns for clarity
    base_cols = ['participant_id', 'alpha_true', 'alpha_hat', 'a_true', 'a_hat', 
                 't0_true', 't0_hat', 'scaler_true', 'scaler_hat']
    
    # Add log_scaler columns if they exist
    if 'log_scaler_true' in df.columns and 'log_scaler_hat' in df.columns:
        base_cols.extend(['log_scaler_true', 'log_scaler_hat'])
    
    col_order = base_cols + ['source_file', 'file_mtime']
    df = df[[col for col in col_order if col in df.columns]]
    
    results_file = outdir / "collected_results.csv"
    df.to_csv(results_file, index=False)
    
    print(f"\n=== Results Summary ===")
    print(f"Collected results from {len(results)} subjects")
    print(f"Results saved to: {results_file}")
    
    # Show parameter recovery metrics if we have truth values
    if truth_df is not None and not df['alpha_true'].isna().all():
        print(f"\n=== Parameter Recovery Metrics ===")
        params_to_check = ['alpha', 'a', 't0', 'scaler']
        
        # Add log_scaler if available
        if 'log_scaler_true' in df.columns and 'log_scaler_hat' in df.columns:
            params_to_check.append('log_scaler')
        
        for param in params_to_check:
            true_col = f'{param}_true'
            est_col = f'{param}_hat'
            
            if true_col in df.columns and est_col in df.columns:
                valid_mask = ~(df[true_col].isna() | df[est_col].isna())
                if valid_mask.sum() > 0:
                    true_vals = df.loc[valid_mask, true_col]
                    est_vals = df.loc[valid_mask, est_col]
                    
                    bias = est_vals.mean() - true_vals.mean()
                    rmse = np.sqrt(((est_vals - true_vals) ** 2).mean())
                    
                    if len(true_vals) > 1:
                        corr = np.corrcoef(true_vals, est_vals)[0, 1]
                    else:
                        corr = np.nan
                    
                    print(f"{param:10s}: n={valid_mask.sum():2d}, bias={bias:6.3f}, rmse={rmse:6.3f}, r={corr:6.3f}")
    
    print(f"\nFirst few results:")
    print(df[['participant_id', 'alpha_true', 'alpha_hat', 'a_true', 'a_hat']].head())

if __name__ == "__main__":
    main()
