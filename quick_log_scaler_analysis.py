#!/usr/bin/env python3
# Quick analysis of log_scaler correlations

import pandas as pd
import numpy as np
import os

print("=== Quick Log Scaler Analysis ===")

# Load the collected results
results_file = "recovery_out/collected_results.csv"
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    print(f"Found {len(df)} results")
    
    # Check if we have log_scaler columns
    if 'log_scaler_true' in df.columns and 'log_scaler_hat' in df.columns:
        # Remove any NaN values
        valid = df.dropna(subset=['log_scaler_true', 'log_scaler_hat'])
        
        if len(valid) > 0:
            # Compute correlations for both raw and log scaler
            scaler_corr = np.corrcoef(valid['scaler_true'], valid['scaler_hat'])[0,1]
            log_scaler_corr = np.corrcoef(valid['log_scaler_true'], valid['log_scaler_hat'])[0,1]
            
            print(f"\n=== Scaler Recovery Comparison ===")
            print(f"Raw scaler correlation:     r = {scaler_corr:.3f}")
            print(f"Log scaler correlation:     r = {log_scaler_corr:.3f}")
            print(f"Improvement with log scale: {log_scaler_corr - scaler_corr:+.3f}")
            
            # Show some examples
            print(f"\nFirst 5 subjects:")
            for i in range(min(5, len(valid))):
                row = valid.iloc[i]
                print(f"Subject {row['participant_id']:2.0f}: scaler {row['scaler_true']:.3f}->{row['scaler_hat']:.3f}, log_scaler {row['log_scaler_true']:.3f}->{row['log_scaler_hat']:.3f}")
        else:
            print("No valid log_scaler data found")
    else:
        print("No log_scaler columns found in results")
        print("Available columns:", list(df.columns))
else:
    print(f"Results file not found: {results_file}")
