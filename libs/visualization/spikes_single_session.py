import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import skew


def quality_metrics(
    df:pd.DataFrame, 
    log_threshold:float=1.0, 
    save_path:str=None
)->None:
    """
    Plots histograms for each numerical column in a DataFrame.
    Histograms are computed first and plotted as line plots with KDE.
    Log scale is applied if a column is highly skewed and contains only positive values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        log_threshold (float): Skewness threshold above which log scale is applied.
    """
    num_cols = df.select_dtypes(include=['number']).columns  # Select numerical columns
    num_plots = len(num_cols)  # Number of subplots
    rows = (num_plots // 3) + (num_plots % 3 > 0)  # Determine subplot rows

    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))  # Create subplots
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, col in enumerate(num_cols):
        data = df[col].dropna()  # Drop NaN values
        data_skewness = skew(data)  # Calculate skewness
        use_log = data_skewness > log_threshold and (data > 0).all()  # Use log scale if highly skewed & positive

        # Define bin edges (normal or log scale)
        if use_log:
            bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 30)  # Log-spaced bins
        else:
            bins = np.linspace(data.min(), data.max(), 30)  # Linearly spaced bins

        # Compute histogram
        counts, bin_edges = np.histogram(data, bins=bins)
    
        # Compute bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot histogram as a line plot
        sns.lineplot(x=bin_centers, y=counts, ax=axes[i], color="blue", lw=2, label="Histogram")

        axes[i].set_title(f"{col}\n(Skewness: {data_skewness:.2f})")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

        # Apply log scale if necessary
        if use_log:
            axes[i].set_xscale("log")
            axes[i].set_title(f"{col} (Log Scale)\n(Skewness: {data_skewness:.2f})")

        axes[i].legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()