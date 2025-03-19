"""
Exploratory script to understand the structure of the NWB data
and identify orientation-related stimulus parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loading import load_nwb_file, extract_epochs_data, extract_neural_data

def explore_epochs_in_detail(nwb):
    """Explore the epochs table in detail to understand stimulus parameters."""
    print("Exploring epochs table in detail...")
    
    # Get epochs table
    epochs = nwb.intervals["epochs"]
    
    # Print all column names
    print(f"Column names: {epochs.colnames}")
    
    # For each column, print more details
    for col_name in epochs.colnames:
        data = epochs[col_name].data[:]
        print(f"\nColumn: {col_name}")
        print(f"  Data type: {data.dtype}")
        print(f"  Shape: {data.shape}")
        
        # If it's a numeric column, print statistics
        if np.issubdtype(data.dtype, np.number):
            print(f"  Min: {np.min(data)}")
            print(f"  Max: {np.max(data)}")
            print(f"  Mean: {np.mean(data)}")
            print(f"  Unique values: {np.unique(data)}")
        
        # If it's a string column, print unique values
        elif data.dtype.kind in ['S', 'U']:
            unique_vals = np.unique(data)
            print(f"  Unique values: {unique_vals}")
        
        # For other types, just print first few values
        else:
            print(f"  First few values: {data[:5]}")
    
    # Try to identify orientation-related columns
    potential_orientation_cols = []
    for col_name in epochs.colnames:
        data = epochs[col_name].data[:]
        
        # Orientation angles are typically between 0 and 360 (or 0 and 180 if non-directional)
        if np.issubdtype(data.dtype, np.number):
            unique_vals = np.unique(data)
            
            # If there are 4, 8, 12, or 16 unique values (common for orientation experiments)
            # and the values are between 0 and 360
            if (len(unique_vals) in [4, 8, 12, 16] and 
                np.min(unique_vals) >= 0 and np.max(unique_vals) <= 360):
                potential_orientation_cols.append(col_name)
                print(f"\nPotential orientation column: {col_name}")
                print(f"  Unique values: {unique_vals}")
    
    return potential_orientation_cols


def explore_stimulus_processing(nwb):
    """Explore the stimulus processing module if available."""
    print("\nExploring stimulus processing module...")
    
    if "stimulus" in nwb.processing:
        stimulus = nwb.processing["stimulus"]
        print(f"Stimulus module contains: {list(stimulus.data_interfaces.keys())}")
        
        # Explore each data interface
        for key in stimulus.data_interfaces:
            interface = stimulus[key]
            print(f"\nInterface: {key}, Type: {type(interface)}")
            
            # If it has data, print info about it
            if hasattr(interface, 'data'):
                data = interface.data[:]
                print(f"  Data shape: {data.shape}")
                print(f"  Data type: {data.dtype}")
                if len(data) > 0:
                    print(f"  First few values: {data[:5]}")
            
            # Also check for timestamps
            if hasattr(interface, 'timestamps'):
                ts = interface.timestamps[:]
                print(f"  Timestamps shape: {ts.shape}")
                if len(ts) > 0:
                    print(f"  First few timestamps: {ts[:5]}")
    else:
        print("No stimulus processing module found.")


def explore_neuron_responses(nwb, data_type="DfOverF"):
    """Explore neural response data to look at the structure."""
    print(f"\nExploring neural responses ({data_type})...")
    
    # First, let's explore the structure of the data in the pipeline
    pipeline = nwb.processing["brain_observatory_pipeline"]
    print(f"Pipeline contains: {list(pipeline.data_interfaces.keys())}")
    
    if "Fluorescence" in pipeline.data_interfaces:
        fluorescence = pipeline["Fluorescence"]
        print(f"Fluorescence container contains: {list(fluorescence.data_interfaces.keys())}")
        
        try:
            # Try to extract the data
            neural_data, timestamps = extract_neural_data(nwb, data_type)
            
            print(f"Neural data shape: {neural_data.shape}")
            print(f"Timestamps shape: {timestamps.shape}")
            
            if len(neural_data) > 0:
                # Get basic stats
                print(f"Data range: [{np.min(neural_data)}, {np.max(neural_data)}]")
                
                # Plot a sample neuron's activity over time
                neuron_idx = 0  # Just look at first neuron for simplicity
                plt.figure(figsize=(12, 5))
                plt.plot(timestamps[:1000], neural_data[:1000, neuron_idx])
                plt.title(f"Sample Neural Activity (Neuron {neuron_idx}, first 1000 samples)")
                plt.xlabel("Time (s)")
                plt.ylabel(f"{data_type} value")
                plt.tight_layout()
                plt.savefig("sample_neural_activity.png")
                plt.close()
                
                print("Saved sample neural activity plot to sample_neural_activity.png")
        except Exception as e:
            print(f"Error extracting neural data: {e}")
    else:
        print("No Fluorescence data found in pipeline")


def identify_trial_structure(nwb):
    """Try to identify how trials are structured in the NWB file."""
    print("\nIdentifying trial structure...")
    
    # Check for epochs
    if "epochs" in nwb.intervals:
        epochs = nwb.intervals["epochs"]
        print(f"Found epochs table with {len(epochs)} entries")
        print(f"Column names: {epochs.colnames}")
    
    # Check for trials
    if "trials" in nwb.intervals:
        trials = nwb.intervals["trials"]
        print(f"Found trials table with {len(trials)} entries")
        print(f"Column names: {trials.colnames}")
    
    # Explore other interval tables if available
    for name in nwb.intervals.keys():
        if name not in ["epochs", "trials"]:
            interval = nwb.intervals[name]
            print(f"Found interval table '{name}' with {len(interval)} entries")
            print(f"Column names: {interval.colnames}")


if __name__ == "__main__":
    # Load the NWB file
    dandiset_id = "000049"
    asset_id = "aa140a7d-cde3-469b-8434-97f13e69c106"
    
    print(f"Loading NWB file from DANDI:{dandiset_id}, asset_id: {asset_id}")
    nwb = load_nwb_file(dandiset_id, asset_id)
    
    # Get basic metadata
    print(f"Session ID: {nwb.identifier}")
    print(f"Subject ID: {nwb.subject.subject_id}")
    print(f"Genotype: {nwb.subject.genotype}")
    
    # Explore the structure
    potential_orientation_cols = explore_epochs_in_detail(nwb)
    explore_stimulus_processing(nwb)
    identify_trial_structure(nwb)
    explore_neuron_responses(nwb)
    
    print("\nData exploration complete!")
    print(f"Potential orientation-related columns: {potential_orientation_cols}")
