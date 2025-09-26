#!/usr/bin/env python3
"""
Gather organized SLURM results from parallel RLDDM fits
Results are organized as: fit_out/{N}state/subject{ID}_*-summary.csv
"""

import pandas as pd
import numpy as np
import glob
import re
from pathlib import Path
from datetime import datetime

def extract_estimates_from_summary(summary_file):
    """Extract parameter estimates from ArviZ summary CSV"""
    try:
        df = pd.read_csv(summary_file, index_col=0)
        
        # Extract key parameters
        params = {}
        for param in ['alpha', 'a', 't0', 'scaler', 'log_scaler']:
            if param in df.index:
                params[f'{param}_mean'] = df.loc[param, 'mean']
                params[f'{param}_sd'] = df.loc[param, 'sd']
                if 'r_hat' in df.columns:
                    params[f'{param}_rhat'] = df.loc[param, 'r_hat']
        
        return params
    except Exception as e:
        print(f"Error processing {summary_file}: {e}")
        return None

def gather_organized_results():
    """Gather results from organized directory structure"""
    
    print("=== Gathering Organized SLURM Results ===")
    
    results = []
    
    # Look for all summary files in organized structure
    summary_files = glob.glob("fit_out/*/subject*-summary.csv")
    print(f"Found {len(summary_files)} summary files")
    
    for summary_file in summary_files:
        # Extract subject ID and states from path and filename
        # Path: fit_out/2state/subject42_rlddm_single_subject-20250925_123456-summary.csv
        
        path_parts = Path(summary_file).parts
        
        # Extract states from directory name (e.g., "2state" -> 2)
        states_match = re.search(r'(\d+)state', path_parts[-2])
        if not states_match:
            print(f"Could not extract states from path: {summary_file}")
            continue
        states = int(states_match.group(1))
        
        # Extract subject ID from filename (e.g., "subject42_" -> 42)
        filename = Path(summary_file).name
        subject_match = re.search(r'subject(\d+)_', filename)
        if not subject_match:
            print(f"Could not extract subject ID from filename: {filename}")
            continue
        subject_id = int(subject_match.group(1))
        
        # Extract parameter estimates
        estimates = extract_estimates_from_summary(summary_file)
        if estimates:
            result = {
                'subject_id': subject_id,
                'states': states,
                'summary_file': summary_file,
                **estimates
            }
            results.append(result)
            print(f"  ✅ Subject {subject_id}, {states} states")
        else:
            print(f"  ❌ Failed to extract estimates: {summary_file}")
    
    if not results:
        print("❌ No results found. Check if SLURM jobs completed successfully.")
        return None
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by states then subject
    df_results = df_results.sort_values(['states', 'subject_id'])
    
    # Save results
    output_file = f"rlddm_slurm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(output_file, index=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"📊 Summary by condition:")
    summary = df_results.groupby('states')['subject_id'].count()
    for states, count in summary.items():
        print(f"  {states} states: {count} subjects")
    
    print(f"\nTotal: {len(df_results)} subjects")
    
    # Display first few results
    print(f"\nFirst 10 results:")
    cols = ['subject_id', 'states', 'alpha_mean', 'a_mean', 't0_mean', 'scaler_mean']
    if all(col in df_results.columns for col in cols):
        print(df_results[cols].head(10).to_string(index=False))
    
    return df_results

if __name__ == "__main__":
    gather_organized_results()
