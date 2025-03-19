"""
Orientation selectivity metrics and analysis functions.

This module provides functions to calculate orientation selectivity metrics
for neural responses to visual stimuli at different orientations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt


def calculate_mean_responses(trials_data: Dict[int, Dict], orientation_column: str = "direction") -> Dict[float, np.ndarray]:
    """
    Calculate mean responses for each orientation/direction.
    
    Parameters
    ----------
    trials_data : Dict[int, Dict]
        Dictionary of trial data with neural responses and stimulus information
    orientation_column : str
        Name of the column containing orientation information
        
    Returns
    -------
    Dict[float, np.ndarray]
        Dictionary mapping orientation angles to mean responses for each neuron
    """
    # Group trials by orientation
    orientation_trials = {}
    
    for trial_id, trial in trials_data.items():
        orientation = trial.get(orientation_column)
        
        # Skip trials with no orientation information
        if orientation is None or np.isnan(orientation):
            continue
        
        if orientation not in orientation_trials:
            orientation_trials[orientation] = []
        
        neural_data = trial.get("neural_data")
        if neural_data is not None and len(neural_data) > 0:
            # Calculate mean response for this trial
            trial_mean = np.mean(neural_data, axis=0)
            orientation_trials[orientation].append(trial_mean)
    
    # Calculate mean response for each orientation
    orientation_responses = {}
    for orientation, responses in orientation_trials.items():
        if responses:  # Check if there are any responses
            orientation_responses[orientation] = np.mean(responses, axis=0)
    
    return orientation_responses


def orientation_selectivity_index(responses: Dict[float, np.ndarray]) -> np.ndarray:
    """
    Calculate the Orientation Selectivity Index (OSI) for each neuron.
    
    OSI = (R_pref - R_orth) / (R_pref + R_orth)
    
    Where:
    - R_pref is the response at the preferred orientation
    - R_orth is the response at the orthogonal orientation (preferred + 90°)
    
    Parameters
    ----------
    responses : Dict[float, np.ndarray]
        Dictionary mapping orientation angles to mean responses
        
    Returns
    -------
    np.ndarray
        OSI values for each neuron
    """
    if not responses:
        return np.array([])
    
    # Get the number of neurons from the first response array
    num_neurons = next(iter(responses.values())).shape[0]
    
    # Find preferred orientation for each neuron
    preferred_orientations = np.zeros(num_neurons)
    max_responses = np.zeros(num_neurons)
    
    # Initialize with the lowest possible value
    max_responses.fill(-np.inf)
    
    for orientation, response in responses.items():
        # For each neuron, check if this orientation gives a higher response
        is_higher = response > max_responses
        
        # Update preferred orientation and max response where applicable
        preferred_orientations = np.where(is_higher, orientation, preferred_orientations)
        max_responses = np.where(is_higher, response, max_responses)
    
    # Calculate orthogonal orientations (preferred + 90 degrees, wrapped to 0-360)
    orthogonal_orientations = (preferred_orientations + 90) % 360
    
    # Get responses at orthogonal orientations
    orthogonal_responses = np.zeros(num_neurons)
    
    # For each neuron, find the closest available orientation to its orthogonal orientation
    for i in range(num_neurons):
        orth_ori = orthogonal_orientations[i]
        
        # Find the closest available orientation
        closest_ori = min(responses.keys(), key=lambda x: min(abs(x - orth_ori), abs(x - (orth_ori + 360))))
        
        orthogonal_responses[i] = responses[closest_ori][i]
    
    # Calculate OSI
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    osi = (max_responses - orthogonal_responses) / (max_responses + orthogonal_responses + epsilon)
    
    return osi


def direction_selectivity_index(responses: Dict[float, np.ndarray]) -> np.ndarray:
    """
    Calculate the Direction Selectivity Index (DSI) for each neuron.
    
    DSI = (R_pref - R_opp) / (R_pref + R_opp)
    
    Where:
    - R_pref is the response at the preferred direction
    - R_opp is the response at the opposite direction (preferred + 180°)
    
    Parameters
    ----------
    responses : Dict[float, np.ndarray]
        Dictionary mapping direction angles to mean responses
        
    Returns
    -------
    np.ndarray
        DSI values for each neuron
    """
    if not responses:
        return np.array([])
    
    # Get the number of neurons from the first response array
    num_neurons = next(iter(responses.values())).shape[0]
    
    # Find preferred direction for each neuron
    preferred_directions = np.zeros(num_neurons)
    max_responses = np.zeros(num_neurons)
    
    # Initialize with the lowest possible value
    max_responses.fill(-np.inf)
    
    for direction, response in responses.items():
        # For each neuron, check if this direction gives a higher response
        is_higher = response > max_responses
        
        # Update preferred direction and max response where applicable
        preferred_directions = np.where(is_higher, direction, preferred_directions)
        max_responses = np.where(is_higher, response, max_responses)
    
    # Calculate opposite directions (preferred + 180 degrees, wrapped to 0-360)
    opposite_directions = (preferred_directions + 180) % 360
    
    # Get responses at opposite directions
    opposite_responses = np.zeros(num_neurons)
    
    # For each neuron, find the closest available direction to its opposite direction
    for i in range(num_neurons):
        opp_dir = opposite_directions[i]
        
        # Find the closest available direction
        closest_dir = min(responses.keys(), key=lambda x: min(abs(x - opp_dir), abs(x - (opp_dir + 360))))
        
        opposite_responses[i] = responses[closest_dir][i]
    
    # Calculate DSI
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    dsi = (max_responses - opposite_responses) / (max_responses + opposite_responses + epsilon)
    
    return dsi


def circular_variance(responses: Dict[float, np.ndarray]) -> np.ndarray:
    """
    Calculate the circular variance (CV) for each neuron.
    
    CV = 1 - |Σ R(θ) * exp(2i*θ)| / Σ R(θ)
    
    where R(θ) is the response at orientation θ.
    
    Parameters
    ----------
    responses : Dict[float, np.ndarray]
        Dictionary mapping orientation angles to mean responses
        
    Returns
    -------
    np.ndarray
        Circular variance values for each neuron (0 = perfectly tuned, 1 = no tuning)
    """
    if not responses:
        return np.array([])
    
    # Get the number of neurons from the first response array
    num_neurons = next(iter(responses.values())).shape[0]
    
    # Initialize numerator and denominator for CV calculation
    sum_vectors = np.zeros(num_neurons, dtype=complex)
    sum_responses = np.zeros(num_neurons)
    
    for orientation, response in responses.items():
        # Convert orientation to radians for circular math
        theta_rad = np.radians(2 * orientation)  # Multiply by 2 for orientations (not directions)
        
        # Calculate complex vectors
        complex_vector = np.exp(1j * theta_rad)
        
        # Add weighted vectors and responses
        sum_vectors += response * complex_vector
        sum_responses += response
    
    # Calculate CV
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    cv = 1 - np.abs(sum_vectors) / (sum_responses + epsilon)
    
    return cv


def plot_tuning_curve(responses: Dict[float, np.ndarray], neuron_idx: int, 
                      osi: Optional[float] = None, dsi: Optional[float] = None, 
                      cv: Optional[float] = None, title: Optional[str] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot orientation tuning curve for a single neuron.
    
    Parameters
    ----------
    responses : Dict[float, np.ndarray]
        Dictionary mapping orientation angles to mean responses
    neuron_idx : int
        Index of the neuron to plot
    osi : float, optional
        Orientation Selectivity Index to display
    dsi : float, optional
        Direction Selectivity Index to display
    cv : float, optional
        Circular Variance to display
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    orientations = np.array(list(responses.keys()))
    neuron_responses = np.array([response[neuron_idx] for response in responses.values()])
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'polar'})
    
    # Convert orientations to radians for polar plot
    theta = np.radians(orientations)
    
    # Repeat the first point at the end to close the curve
    theta = np.append(theta, theta[0])
    neuron_responses = np.append(neuron_responses, neuron_responses[0])
    
    # Plot the tuning curve
    ax.plot(theta, neuron_responses, 'o-', linewidth=2)
    
    # Set up the plot
    ax.set_theta_zero_location('N')  # 0 degrees at the top
    ax.set_theta_direction(-1)  # Clockwise
    
    # Set up the title with metrics if provided
    plot_title = title or f"Orientation Tuning Curve for Neuron {neuron_idx}"
    metrics_text = ""
    
    if osi is not None:
        metrics_text += f"OSI: {osi:.3f} "
    if dsi is not None:
        metrics_text += f"DSI: {dsi:.3f} "
    if cv is not None:
        metrics_text += f"CV: {cv:.3f}"
    
    if metrics_text:
        plot_title += f"\n{metrics_text}"
    
    ax.set_title(plot_title)
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def plot_selectivity_distribution(osi_values: np.ndarray, title: str = "Distribution of Orientation Selectivity Index",
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of orientation selectivity indices across neurons.
    
    Parameters
    ----------
    osi_values : np.ndarray
        Array of OSI values for each neuron
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(osi_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add statistics
    median_osi = np.median(osi_values)
    mean_osi = np.mean(osi_values)
    
    ax.axvline(median_osi, color='red', linestyle='--', linewidth=2, label=f'Median: {median_osi:.3f}')
    ax.axvline(mean_osi, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_osi:.3f}')
    
    # Set up the plot
    ax.set_xlabel('Orientation Selectivity Index')
    ax.set_ylabel('Number of Neurons')
    ax.set_title(title)
    ax.legend()
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def find_best_orientation_neuron(osi_values: np.ndarray, responses: Dict[float, np.ndarray]) -> Tuple[int, float]:
    """
    Find the neuron with the highest orientation selectivity.
    
    Parameters
    ----------
    osi_values : np.ndarray
        Array of OSI values for each neuron
    responses : Dict[float, np.ndarray]
        Dictionary mapping orientation angles to mean responses
        
    Returns
    -------
    Tuple[int, float]
        Index of the neuron with highest OSI and its OSI value
    """
    # Find the neuron with highest OSI
    best_neuron_idx = np.argmax(osi_values)
    best_osi = osi_values[best_neuron_idx]
    
    return best_neuron_idx, best_osi


if __name__ == "__main__":
    # Example usage with synthetic data
    num_neurons = 10
    orientations = [0, 45, 90, 135, 180, 225, 270, 315]
    
    # Create fake responses with some neurons having orientation preference
    responses = {}
    for ori in orientations:
        # Generate random responses with some neurons having preference
        resp = np.random.rand(num_neurons) * 0.5
        
        # Make some neurons selective for specific orientations
        for i in range(num_neurons):
            preferred_ori = (i * 45) % 360  # Each neuron prefers a different orientation
            # Higher response at preferred orientation, drops off with angular distance
            resp[i] += np.exp(-np.min([abs(ori - preferred_ori), abs(ori - preferred_ori + 360), 
                                       abs(ori - preferred_ori - 360)]) / 45.0)
        
        responses[ori] = resp
    
    # Calculate OSI
    osi_values = orientation_selectivity_index(responses)
    
    # Find best neuron
    best_neuron, best_osi = find_best_orientation_neuron(osi_values, responses)
    
    # Plot tuning curve for best neuron
    plot_tuning_curve(responses, best_neuron, best_osi, title=f"Best Orientation Tuned Neuron (#{best_neuron})")
    
    # Plot OSI distribution
    plot_selectivity_distribution(osi_values)
    
    plt.show()
