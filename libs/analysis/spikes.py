import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import csv
import pickle
from scipy.ndimage import gaussian_filter1d



def compute_peth_array(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    bin_size: float = 0.05,
    time_window: list = [-1, 1],
    plot: bool = True
) -> dict:
    """
    Computes Peri-Event Time Histogram (PETH) per trial without averaging.
    
    Parameters:
        spiketime (numpy array): Spike times in seconds.
        eventtime (numpy array): Event times in seconds.
        bin_size (float): Bin width in seconds.
        time_window (list): Time window around events [pre_time, post_time].
    
    Returns:
        bins (numpy array): Bin edges (time axis).
        trial_peth (numpy array): 2D array (trials × time bins) with spike counts.
        mean_firing_rate (numpy array): Mean firing rate across trials (Hz).
    """
    pre, post = time_window  # Define pre and post event time window
    bins = np.arange(pre, post + bin_size, bin_size)  # Create time bins
    n_trials = len(event_times)  # Number of trials (events)
    n_bins = len(bins) - 1  # Number of bins
    spike_counts = np.zeros((n_trials, n_bins))  # Initialize spike count array
    trial = np.zeros(n_trials)  # Initialize trial index array
    
    # Compute spike counts for each trial
    for i_trial, event in enumerate(event_times):
        trial[i_trial] = i_trial  # Store trial index
        aligned_spikes = spike_times - event  # Align spike times to event time
        valid_spikes = aligned_spikes[(aligned_spikes >= pre) & (aligned_spikes <= post)]  # Keep spikes within time window
        hist, _ = np.histogram(valid_spikes, bins=bins)  # Compute histogram for spike counts
        spike_counts[i_trial, :] = hist  # Store spike counts for each trial
    spike_counts = spike_counts.astype(np.uint8) 

    # Compute mean firing rate (Hz)
    mean_firing_rate = spike_counts.mean(axis=0)/bin_size

    # Plot PETH and mean firing rate
    if plot is True:
        fig = plt.figure(figsize=(6, 8))  # Set the overall figure size
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[3, 1], wspace=0.05, hspace=0.1)

        # **Heatmap of spike counts per trial**
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(spike_counts, aspect='auto', cmap='gray_r', extent=[pre, post, 0, n_trials])
        ax1.axvline(0, color='red', linestyle='dashed')  # Mark event time (t=0)
        ax1.set_ylabel("Trial")
        ax1.set_title("Peri-event time histogram (PETH)")

        # **Place the colorbar on the right side**
        cbar_ax = fig.add_subplot(gs[0, 1])  # Create a subplot for the colorbar
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Spike count")

        # **Mean firing rate across trials**
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.plot(bins[:-1], mean_firing_rate, color="black", lw=2)
        ax2.axvline(0, color='red', linestyle='dashed')  # Mark event time (t=0)
        ax2.set_xlabel("Time from event (s)")
        ax2.set_ylabel("Spikes/sec")

        plt.show()
    
    
    bins = np.array(bins)# Convert bins to numpy array
    bins = np.round(bins, 5)  # Round bins to 3 decimal places
    
    trial = np.array(trial)  # Convert trial indices to numpy array
    trial = trial.astype(int)  # Convert trial indices to integers
    
    spike_counts = np.array(spike_counts)  # Convert spike counts to numpy array
    spike_counts = spike_counts.astype(np.uint8)  # Convert spike counts to integers
    
    peth = {'trial':trial, 'bins': bins, 'spike_counts': spike_counts}

    return peth



def compute_peth_smooth(
    peth: dict,
    sigma_sec: float = 0.02,
    ) -> dict:
    
    # Load PETH data
    trial = peth['trial']
    bins = peth['bins']
    spike_counts = peth['spike_counts'].astype(float)

    # Compute filter parameters
    bin_size_sec=np.diff(bins).mean()
    sigma_bins=sigma_sec/bin_size_sec

    spike_density = gaussian_filter1d(spike_counts, sigma=sigma_bins, axis=1, mode='constant')

    peth = {'trial':trial, 'bins': bins, 'spike_density': spike_density}
    
    return peth
    
    


def compute_peth_df(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    bin_size: float = 0.05,
    time_window: list = [-1, 1],
    plot: bool = True
) -> pd.DataFrame:
    """
    Computes PETH (Peri-Event Time Histogram) per trial and returns a long-format DataFrame.

    Returns:
        DataFrame with columns: 'Time', 'Trial', 'Spike_count'
    """
    pre, post = time_window
    bins = np.arange(pre, post + bin_size, bin_size)
    n_trials = len(event_times)
    n_bins = len(bins) - 1
    trial_spike_counts = np.zeros((n_trials, n_bins))

    for i, event in enumerate(event_times):
        aligned_spikes = spike_times - event
        valid_spikes = aligned_spikes[(aligned_spikes >= pre) & (aligned_spikes <= post)]
        hist, _ = np.histogram(valid_spikes, bins=bins)
        trial_spike_counts[i, :] = hist

    trial_spike_rate = trial_spike_counts.astype(np.uint8) / bin_size
    mean_firing_rate = trial_spike_counts.mean(axis=0) / bin_size

    # Plot
    if plot is True:
        fig = plt.figure(figsize=(6, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[3, 1], wspace=0.05, hspace=0.1)

        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(trial_spike_counts, aspect='auto', cmap='gray_r', extent=[pre, post, 0, n_trials])
        ax1.axvline(0, color='red', linestyle='dashed')
        ax1.set_ylabel("Trial")
        ax1.set_title("Peri-event time histogram (PETH)")

        cbar_ax = fig.add_subplot(gs[0, 1])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Spike count")

        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.plot(bins[:-1], mean_firing_rate, color="black", lw=2)
        ax2.axvline(0, color='red', linestyle='dashed')
        ax2.set_xlabel("Time from event (s)")
        ax2.set_ylabel("Spikes/sec")

        plt.show()

    # Create long-format DataFrame
    time_points = bins[:-1]
    df_list = []
    for trial_idx in range(n_trials):
        df_list.append(pd.DataFrame({
            'Time': time_points,
            'Trial': trial_idx,
            'Spike_rate': trial_spike_rate[trial_idx]
        }))
    
    peth = pd.concat(df_list, ignore_index=True)
    
    return peth
