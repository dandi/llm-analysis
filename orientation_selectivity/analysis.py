"""
Main analysis script for orientation selectivity.

This script performs the full analysis workflow:
1. Load NWB data and extract neural responses by stimulus orientation
2. Calculate orientation selectivity metrics for each neuron
3. Find the most selective neurons
4. Generate and save tuning curves and other figures
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loading import load_nwb_file, extract_epochs_data, extract_neural_data, extract_trial_oriented_data
from selectivity import (
    calculate_mean_responses,
    orientation_selectivity_index,
    direction_selectivity_index,
    circular_variance,
    plot_tuning_curve,
    plot_selectivity_distribution,
    find_best_orientation_neuron
)


def create_output_dirs(base_dir="."):
    """Create output directories for results if they don't exist."""
    figures_dir = Path(base_dir) / "figures"
    results_dir = Path(base_dir) / "results"
    
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    return figures_dir, results_dir


def process_orientation_data(nwb, data_type="DfOverF"):
    """
    Process orientation selectivity data from NWB file.
    
    Parameters
    ----------
    nwb : pynwb.NWBFile
        NWB file object
    data_type : str
        Type of neural data to analyze (e.g., 'DfOverF', 'demixed_traces')
        
    Returns
    -------
    dict
        Dictionary containing processed data and results
    """
    print(f"Processing orientation data with {data_type} signals...")
    
    # Create output directories
    figures_dir, results_dir = create_output_dirs()
    
    # Extract epoch data (stimulus information)
    epochs = extract_epochs_data(nwb)
    print(f"Extracted epochs data with keys: {list(epochs.keys())}")
    
    # Try different data types if DfOverF doesn't work
    data_types_to_try = ["DfOverF", "demixed_traces", "raw_traces", "dff_events"]
    neural_data = np.array([])
    timestamps = np.array([])
    
    for dt in data_types_to_try:
        print(f"Trying to extract {dt}...")
        neural_data, timestamps = extract_neural_data(nwb, dt)
        if len(neural_data) > 0:
            print(f"Successfully extracted {dt} data!")
            data_type = dt
            break
    
    if len(neural_data) == 0:
        print("ERROR: Could not extract neural data with any of the known data types.")
        print("Please check the structure of the NWB file.")
        return {}
    
    print(f"Extracted neural data: shape {neural_data.shape}")
    
    # Prepare trial-oriented data
    trials_data = {}
    
    # Get the number of epochs
    num_epochs = len(epochs["start_time"])
    
    for i in range(num_epochs):
        start_time = epochs["start_time"][i]
        stop_time = epochs["stop_time"][i]
        
        # Skip trials where direction is NaN
        if np.isnan(epochs["direction"][i]):
            continue
        
        # Find indices of timestamps that fall within this trial
        trial_mask = (timestamps >= start_time) & (timestamps <= stop_time)
        
        # Store trial data
        trials_data[i] = {
            "start_time": start_time,
            "stop_time": stop_time,
            "neural_data": neural_data[trial_mask] if len(neural_data) > 0 else [],
            "timestamps": timestamps[trial_mask] if len(timestamps) > 0 else [],
            "direction": epochs["direction"][i],
            "temporal_frequency": epochs["temporal_frequency"][i] if "temporal_frequency" in epochs else None,
            "spatial_frequency": epochs["spatial_frequency"][i] if "spatial_frequency" in epochs else None,
            "contrast": epochs["contrast"][i] if "contrast" in epochs else None
        }
    
    # Calculate mean responses for each orientation
    direction_responses = calculate_mean_responses(trials_data, orientation_column="direction")
    
    # For this dataset, directions also serve as orientations (treating 0° and 180° as same orientation)
    print(f"Found responses for {len(direction_responses)} directions: {list(direction_responses.keys())}")
    
    # Calculate orientation and direction selectivity metrics
    osi_values = orientation_selectivity_index(direction_responses)
    dsi_values = direction_selectivity_index(direction_responses) 
    cv_values = circular_variance(direction_responses)
    
    # Find the best orientation tuned neuron
    best_neuron_idx, best_osi = find_best_orientation_neuron(osi_values, direction_responses)
    
    print(f"Best orientation-tuned neuron: #{best_neuron_idx} (OSI: {best_osi:.3f})")
    
    # Plot and save tuning curve for the best neuron
    best_neuron_fig = plot_tuning_curve(
        direction_responses, 
        best_neuron_idx, 
        osi=osi_values[best_neuron_idx],
        dsi=dsi_values[best_neuron_idx],
        cv=cv_values[best_neuron_idx],
        title=f"Orientation Tuning Curve for Best Neuron (#{best_neuron_idx})",
        save_path=figures_dir / "best_neuron_tuning.png"
    )
    
    # Plot and save distribution of OSI values
    osi_dist_fig = plot_selectivity_distribution(
        osi_values,
        title="Distribution of Orientation Selectivity Index",
        save_path=figures_dir / "osi_distribution.png"
    )
    
    # Save numeric results
    np.savez(
        results_dir / "orientation_selectivity_results.npz",
        osi_values=osi_values,
        dsi_values=dsi_values,
        cv_values=cv_values,
        best_neuron_idx=best_neuron_idx
    )
    
    # Collect results
    results = {
        "osi_values": osi_values,
        "dsi_values": dsi_values,
        "cv_values": cv_values,
        "best_neuron_idx": best_neuron_idx,
        "direction_responses": direction_responses,
        "figures": {
            "best_neuron_tuning": best_neuron_fig,
            "osi_distribution": osi_dist_fig
        }
    }
    
    return results


def analyze_allen_visual_cortex_data(dandiset_id, asset_id):
    """
    Analyze orientation selectivity in a specific Allen Brain Observatory dataset.
    
    Parameters
    ----------
    dandiset_id : str
        DANDI dataset ID
    asset_id : str
        Asset ID for the specific NWB file
    
    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    print(f"Analyzing orientation selectivity in DANDI:{dandiset_id}, asset: {asset_id}")
    
    # Load the NWB file
    nwb = load_nwb_file(dandiset_id, asset_id)
    
    # Display basic metadata
    print(f"Session ID: {nwb.identifier}")
    print(f"Subject ID: {nwb.subject.subject_id}")
    print(f"Genotype: {nwb.subject.genotype}")
    
    # Process the data
    results = process_orientation_data(nwb, data_type="DfOverF")
    
    print("Analysis complete!")
    print(f"Found {len(results['osi_values'])} neurons with median OSI: {np.median(results['osi_values']):.3f}")
    
    return results


if __name__ == "__main__":
    # Allen Brain Observatory dataset
    dandiset_id = "000049"
    asset_id = "aa140a7d-cde3-469b-8434-97f13e69c106"
    
    # Run the analysis
    results = analyze_allen_visual_cortex_data(dandiset_id, asset_id)
    
    # Show the figures (if running interactively)
    plt.show()
