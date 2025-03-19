"""
Data loading utilities for orientation selectivity analysis.

This module provides functions to load and process data from NWB files
containing visual cortex recordings with orientation tuning experiments.
"""

import numpy as np
import pynwb
import lindi
from typing import Dict, List, Tuple, Optional, Any


def load_nwb_file(dandiset_id: str, asset_id: str) -> pynwb.NWBFile:
    """
    Load an NWB file from DANDI archive using lindi.
    
    Parameters
    ----------
    dandiset_id : str
        DANDI dataset ID (e.g., '000049')
    asset_id : str
        Asset ID for the specific NWB file
        
    Returns
    -------
    pynwb.NWBFile
        The loaded NWB file object
    """
    lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/{dandiset_id}/assets/{asset_id}/nwb.lindi.json"
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
    return nwb


def extract_epochs_data(nwb: pynwb.NWBFile) -> Dict[str, np.ndarray]:
    """
    Extract epoch data from the NWB file.
    
    Parameters
    ----------
    nwb : pynwb.NWBFile
        The NWB file object
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing epoch information
    """
    epochs = nwb.intervals["epochs"]
    
    # Extract available fields
    data = {}
    for field_name in epochs.colnames:
        data[field_name] = epochs[field_name].data[:]
    
    return data


def extract_stimulus_data(nwb: pynwb.NWBFile) -> Dict[str, np.ndarray]:
    """
    Extract stimulus data from the NWB file.
    
    Parameters
    ----------
    nwb : pynwb.NWBFile
        The NWB file object
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing stimulus information
    """
    if "stimulus" not in nwb.processing:
        return {}
    
    stimulus_module = nwb.processing["stimulus"]
    
    # Extract available data
    data = {}
    for key in stimulus_module.data_interfaces:
        interface = stimulus_module[key]
        if hasattr(interface, 'data'):
            data[key] = interface.data[:]
        # Also extract timestamps if available
        if hasattr(interface, 'timestamps'):
            data[f"{key}_timestamps"] = interface.timestamps[:]
    
    return data


def extract_neural_data(nwb: pynwb.NWBFile, data_type: str = "DfOverF") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract neural activity data from the NWB file.
    
    Parameters
    ----------
    nwb : pynwb.NWBFile
        The NWB file object
    data_type : str
        Type of neural data to extract ('DfOverF', 'demixed_traces', etc.')
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Neural activity data and corresponding timestamps
    """
    if "brain_observatory_pipeline" not in nwb.processing:
        return np.array([]), np.array([])
    
    pipeline = nwb.processing["brain_observatory_pipeline"]
    
    # Check if fluorescence data exists
    if "Fluorescence" not in pipeline.data_interfaces:
        return np.array([]), np.array([])
    
    fluorescence = pipeline["Fluorescence"]
    
    # Check if the requested data type exists
    try:
        # Try to access as a dictionary key first
        data_series = fluorescence[data_type]
        return data_series.data[:], data_series.timestamps[:]
    except (KeyError, TypeError, AttributeError):
        # If that fails, try alternate ways to access the data
        try:
            # For pynwb.ophys.Fluorescence object, try using get_roi_response_series method
            if hasattr(fluorescence, 'get_roi_response_series'):
                all_series = fluorescence.roi_response_series
                if data_type in all_series:
                    data_series = all_series[data_type]
                    return data_series.data[:], data_series.timestamps[:]
            
            # If we can get a list of available types, try to find the best match
            if hasattr(fluorescence, 'roi_response_series'):
                # If data_type isn't available, but we have data, use the first available
                if fluorescence.roi_response_series:
                    first_key = next(iter(fluorescence.roi_response_series.keys()))
                    print(f"Data type '{data_type}' not found, using '{first_key}' instead")
                    data_series = fluorescence.roi_response_series[first_key]
                    return data_series.data[:], data_series.timestamps[:]
        except Exception as e:
            print(f"Error accessing neural data: {e}")
            pass
    
    # If all attempts fail, return empty arrays
    return np.array([]), np.array([])


def get_metadata(nwb: pynwb.NWBFile) -> Dict[str, Any]:
    """
    Extract metadata from the NWB file.
    
    Parameters
    ----------
    nwb : pynwb.NWBFile
        The NWB file object
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing metadata
    """
    metadata = {
        "session_id": nwb.identifier,
        "session_start_time": nwb.session_start_time,
        "subject_id": nwb.subject.subject_id,
        "genotype": nwb.subject.genotype,
        "sex": nwb.subject.sex,
        "age": nwb.subject.age,
    }
    
    return metadata


def explore_epoch_structure(nwb: pynwb.NWBFile) -> Dict[str, Any]:
    """
    Explore the structure of epoch data to understand stimulus parameters.
    
    Parameters
    ----------
    nwb : pynwb.NWBFile
        The NWB file object
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing epoch structure details
    """
    epochs = nwb.intervals["epochs"]
    
    # Get column names to understand the structure
    column_info = {}
    for col_name in epochs.colnames:
        data = epochs[col_name].data[:]
        # Get unique values to understand the range of values in this column
        unique_vals = np.unique(data)
        column_info[col_name] = {
            "data_type": str(data.dtype),
            "shape": data.shape,
            "unique_values": unique_vals if len(unique_vals) <= 20 else f"{len(unique_vals)} unique values"
        }
    
    return column_info


def extract_trial_oriented_data(nwb: pynwb.NWBFile, data_type: str = "DfOverF") -> Dict[int, Dict[str, Any]]:
    """
    Extract neural data organized by trials with stimulus information.
    This is a placeholder that will be expanded once we understand the trial structure.
    
    Parameters
    ----------
    nwb : pynwb.NWBFile
        The NWB file object
    data_type : str
        Type of neural data to extract
        
    Returns
    -------
    Dict[int, Dict[str, Any]]
        Dictionary of trial data with neural responses and stimulus information
    """
    # This is a placeholder that will be expanded once we understand the epoch structure
    epochs = extract_epochs_data(nwb)
    neural_data, timestamps = extract_neural_data(nwb, data_type)
    
    # For now, just return a basic structure
    trials_data = {}
    
    # Get the number of epochs
    num_epochs = len(epochs["start_time"])
    
    for i in range(num_epochs):
        start_time = epochs["start_time"][i]
        stop_time = epochs["stop_time"][i]
        
        # Find indices of timestamps that fall within this trial
        trial_mask = (timestamps >= start_time) & (timestamps <= stop_time)
        
        # Store trial data
        trials_data[i] = {
            "start_time": start_time,
            "stop_time": stop_time,
            "neural_data": neural_data[trial_mask] if len(neural_data) > 0 else [],
            "timestamps": timestamps[trial_mask] if len(timestamps) > 0 else [],
            # Stimulus parameters will be added once we understand the structure
        }
    
    return trials_data


if __name__ == "__main__":
    # Example usage
    dandiset_id = "000049"
    asset_id = "aa140a7d-cde3-469b-8434-97f13e69c106"
    
    # Load the NWB file
    nwb = load_nwb_file(dandiset_id, asset_id)
    
    # Explore the epoch structure to understand how stimulus parameters are stored
    epoch_structure = explore_epoch_structure(nwb)
    print("Epoch structure:")
    for col_name, info in epoch_structure.items():
        print(f"  {col_name}: {info}")
    
    # Extract epoch data
    epochs = extract_epochs_data(nwb)
    print(f"\nFound {len(epochs['start_time'])} epochs")
    
    # Extract neural data
    neural_data, timestamps = extract_neural_data(nwb)
    print(f"\nNeural data shape: {neural_data.shape}")
    print(f"Timestamps shape: {timestamps.shape}")
