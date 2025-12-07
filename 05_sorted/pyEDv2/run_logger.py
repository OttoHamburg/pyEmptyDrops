#!/usr/bin/env python3
"""
Run logging utilities for pyEDv2 validation.

This module provides functions to log pyEDv2 run metadata to a CSV file
for tracking performance and accuracy across different parameter settings.
"""

import pandas as pd
import os
from pathlib import Path

def log_run_metadata(metadata_dict, log_file_path="runs/run_log.csv"):
    """
    Log run metadata to a CSV file.
    
    Parameters
    ----------
    metadata_dict : dict
        Dictionary containing run metadata
    log_file_path : str
        Path to the log CSV file (relative to pyEDv2 directory)
    """
    # Ensure the directory exists
    log_file = Path(log_file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert metadata to DataFrame
    df_new = pd.DataFrame([metadata_dict])
    
    # Check if log file exists
    if log_file.exists():
        # Read existing log and append
        df_existing = pd.read_csv(log_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # Create new log file with header
        df_combined = df_new
    
    # Save to CSV
    df_combined.to_csv(log_file, index=False)
    print(f"Run metadata logged to: {log_file}")
    
    return str(log_file)

def save_results_to_runs_folder(results_df, metadata_dict, runs_folder="runs"):
    """
    Save results DataFrame to the runs folder with timestamp.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        EmptyDrops results DataFrame
    metadata_dict : dict
        Run metadata dictionary
    runs_folder : str
        Path to runs folder
        
    Returns
    -------
    str
        Path to saved results file
    """
    # Ensure runs folder exists
    runs_path = Path(runs_folder)
    runs_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp and key parameters
    timestamp = metadata_dict['timestamp']
    niters = metadata_dict['niters']
    filename = f"niters{niters}_results_{timestamp}.csv"
    
    # Save results
    results_file = runs_path / filename
    results_df.to_csv(results_file)
    
    print(f"Results saved to: {results_file}")
    return str(results_file)

def get_run_summary(log_file_path="runs/run_log.csv"):
    """
    Get a summary of all logged runs.
    
    Parameters
    ----------
    log_file_path : str
        Path to the log CSV file
        
    Returns
    -------
    pd.DataFrame or None
        Summary DataFrame or None if no log exists
    """
    log_file = Path(log_file_path)
    
    if not log_file.exists():
        print(f"No log file found at: {log_file}")
        return None
    
    df = pd.read_csv(log_file)
    
    return df

def compare_with_r_targets(metadata_dict):
    """
    Compare run results with R targets and return assessment.
    
    Parameters
    ----------
    metadata_dict : dict
        Run metadata dictionary
        
    Returns
    -------
    dict
        Assessment results
    """
    # R reference data
    r_reference = {
        50: {'fdr_0.001': 516, 'fdr_0.01': 516, 'fdr_0.05': 7597, 'runtime_sec': 2.5},
        500: {'fdr_0.001': 516, 'fdr_0.01': 7469, 'fdr_0.05': 8267, 'runtime_sec': 3.9},
        10000: {'fdr_0.001': 6892, 'fdr_0.01': 7505, 'fdr_0.05': 8182, 'runtime_sec': 31.0}
    }
    
    niters = metadata_dict['niters']
    
    # Find closest R reference
    if niters in r_reference:
        r_ref = r_reference[niters]
        match_type = "exact"
    else:
        available_niters = list(r_reference.keys())
        closest_niters = min(available_niters, key=lambda x: abs(x - niters))
        r_ref = r_reference[closest_niters]
        match_type = f"closest (using {closest_niters})"
    
    # Calculate differences
    diff_001 = metadata_dict['fdr_0_001'] - r_ref['fdr_0.001']
    diff_01 = metadata_dict['fdr_0_01'] - r_ref['fdr_0.01']
    diff_05 = metadata_dict['fdr_0_05'] - r_ref['fdr_0.05']
    
    # Calculate percentages
    pct_001 = (diff_001 / r_ref['fdr_0.001']) * 100 if r_ref['fdr_0.001'] > 0 else 0
    pct_01 = (diff_01 / r_ref['fdr_0.01']) * 100 if r_ref['fdr_0.01'] > 0 else 0
    pct_05 = (diff_05 / r_ref['fdr_0.05']) * 100 if r_ref['fdr_0.05'] > 0 else 0
    
    # Success criteria
    knee_success = metadata_dict['calculated_retain'] == 10754
    success_001 = abs(diff_001) <= 5
    success_01 = abs(diff_01) <= 5
    success_05 = abs(pct_05) <= 5.0
    
    assessment = {
        'r_reference': r_ref,
        'match_type': match_type,
        'differences': {
            'fdr_0_001': diff_001,
            'fdr_0_01': diff_01,
            'fdr_0_05': diff_05
        },
        'percentages': {
            'fdr_0_001': pct_001,
            'fdr_0_01': pct_01,
            'fdr_0_05': pct_05
        },
        'success_flags': {
            'knee_detection': knee_success,
            'fdr_0_001': success_001,
            'fdr_0_01': success_01,
            'fdr_0_05': success_05
        },
        'overall_success': knee_success and success_001 and success_01 and success_05
    }
    
    return assessment