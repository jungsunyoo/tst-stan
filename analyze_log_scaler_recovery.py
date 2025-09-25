#!/usr/bin/env python3
# analyze_log_scaler_recovery.py
# Quick analysis focusing on log_scaler parameter recovery

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

def find_and_analyze_results():
    """Find results and analyze log_scaler recovery specifically"""
    
    print("=== Log Scaler Parameter Recovery Analysis ===")
    
    # Load truth parameters
    truth_file = "recovery_out/truth_params.csv"
    if not os.path.exists(truth_file):
        print(f"❌ Truth file not found: {truth_file}")
        return
        
    truth_df = pd.read_csv(truth_file)
    print(f"📊 Found truth parameters for {len(truth_df)} subjects")
    print(f"Truth file columns: {list(truth_df.columns)}")
    
    # Find summary files 
    patterns = [
        "recovery_out/**/rlddm_single_subject-*-summary.csv",
        "stan_out/*-summary.csv",
        "cmdstan_tmp/**/*-summary.csv"
    ]
    
    summary_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        summary_files.extend(files)
    
    print(f"Found {len(summary_files)} potential summary files")
    
    # Extract parameters from summary files
    results = []
    for i, summary_file in enumerate(summary_files[:20]):  # Limit to first 20 for our analysis
        try:
            df = pd.read_csv(summary_file, index_col=0)
            
            # Check if this has our parameters
            required = ['alpha', 'a', 't0', 'scaler']
            if not all(param in df.index for param in required):
                continue
                
            result = {
                'subject_idx': i + 1,  # Sequential assignment for now
                'alpha_hat': float(df.loc['alpha', 'mean']),
                'a_hat': float(df.loc['a', 'mean']),
                't0_hat': float(df.loc['t0', 'mean']),
                'scaler_hat': float(df.loc['scaler', 'mean']),
                'source_file': summary_file
            }
            
            # Add log_scaler if available
            if 'log_scaler' in df.index:
                result['log_scaler_hat'] = float(df.loc['log_scaler', 'mean'])
                print(f"✅ Subject {i+1}: Found log_scaler = {result['log_scaler_hat']:.3f}")
            else:
                print(f"⚠️  Subject {i+1}: No log_scaler found")
                result['log_scaler_hat'] = np.nan
                
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing {summary_file}: {e}")
            
    if not results:
        print("❌ No valid results found")
        return
        
    print(f"\n✅ Successfully extracted {len(results)} results")
    
    # Match with truth data
    analysis_data = []
    for result in results:
        subj_id = result['subject_idx']
        if subj_id <= len(truth_df):
            truth_row = truth_df.iloc[subj_id - 1]  # 0-based indexing
            
            analysis_data.append({
                'participant_id': subj_id,
                'alpha_true': truth_row['alpha_true'],
                'alpha_hat': result['alpha_hat'],
                'a_true': truth_row['a_true'],
                'a_hat': result['a_hat'],
                't0_true': truth_row['t0_true'],
                't0_hat': result['t0_hat'],
                'scaler_true': truth_row['scaler_true'],
                'scaler_hat': result['scaler_hat'],
                'log_scaler_true': truth_row['log_scaler_true'],
                'log_scaler_hat': result.get('log_scaler_hat', np.nan),
                'source_file': result['source_file']
            })
    
    if not analysis_data:
        print("❌ No matched data found")
        return
        
    df = pd.DataFrame(analysis_data)
    
    # Save results
    output_file = "recovery_out/log_scaler_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\n📁 Results saved to: {output_file}")
    
    # Analyze correlations
    print(f"\n=== Parameter Recovery Analysis (n={len(df)}) ===")
    
    params = ['alpha', 'a', 't0', 'scaler', 'log_scaler']
    for param in params:
        true_col = f'{param}_true'
        est_col = f'{param}_hat'
        
        if true_col in df.columns and est_col in df.columns:
            # Remove NaN values
            valid_mask = ~(df[true_col].isna() | df[est_col].isna())
            n_valid = valid_mask.sum()
            
            if n_valid > 1:
                true_vals = df.loc[valid_mask, true_col]
                est_vals = df.loc[valid_mask, est_col]
                
                bias = est_vals.mean() - true_vals.mean()
                rmse = np.sqrt(((est_vals - true_vals) ** 2).mean())
                corr = np.corrcoef(true_vals, est_vals)[0, 1]
                
                print(f"{param:10s}: n={n_valid:2d}, bias={bias:+7.3f}, rmse={rmse:7.3f}, r={corr:7.3f}")
                
                # Show individual comparisons for log_scaler
                if param == 'log_scaler' and n_valid > 0:
                    print(f"\nLog Scaler Individual Comparisons:")
                    for idx, row in df[valid_mask].iterrows():
                        true_val = row[true_col]
                        est_val = row[est_col]
                        error = est_val - true_val
                        print(f"  Subject {row['participant_id']:2d}: true={true_val:6.3f}, est={est_val:6.3f}, error={error:+6.3f}")
            else:
                print(f"{param:10s}: n={n_valid:2d}, insufficient data")
    
    # Compare raw scaler vs log_scaler correlations
    if ('scaler_true' in df.columns and 'scaler_hat' in df.columns and 
        'log_scaler_true' in df.columns and 'log_scaler_hat' in df.columns):
        
        # Raw scaler correlation
        valid_raw = ~(df['scaler_true'].isna() | df['scaler_hat'].isna())
        if valid_raw.sum() > 1:
            r_raw = np.corrcoef(df.loc[valid_raw, 'scaler_true'], 
                               df.loc[valid_raw, 'scaler_hat'])[0, 1]
        else:
            r_raw = np.nan
            
        # Log scaler correlation  
        valid_log = ~(df['log_scaler_true'].isna() | df['log_scaler_hat'].isna())
        if valid_log.sum() > 1:
            r_log = np.corrcoef(df.loc[valid_log, 'log_scaler_true'], 
                               df.loc[valid_log, 'log_scaler_hat'])[0, 1]
        else:
            r_log = np.nan
            
        print(f"\n=== Scaler Parameter Comparison ===")
        print(f"Raw scaler correlation (r):     {r_raw:7.3f}")
        print(f"Log scaler correlation (r):     {r_log:7.3f}")
        print(f"Improvement in correlation:     {r_log - r_raw:+7.3f}")
        
        if r_log > r_raw:
            print("✅ Log-scale analysis shows better parameter recovery!")
        else:
            print("⚠️  Raw scale shows better correlation")

if __name__ == "__main__":
    find_and_analyze_results()
