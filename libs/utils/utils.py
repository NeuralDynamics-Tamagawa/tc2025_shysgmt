import numpy as np
import pandas as pd
from datetime import datetime
from IPython import get_ipython
import os
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, List, Literal


# Figure setting for biology
def biofig(ax=None, linewidth=2, fontsize=12, grid=False, hide_top=True, hide_right=True):
    """
    Customizes the appearance of matplotlib axes.

    Parameters:
        ax (matplotlib.axes.Axes, optional): The target axis to apply the settings. Defaults to the current axis.
        linewidth (int, optional): Thickness of the left and bottom spines. Default is 2.
        fontsize (int, optional): Font size for labels, titles, and tick labels. Default is 12.
        grid (bool, optional): Whether to display the grid. Default is False.
        hide_top (bool, optional): Whether to hide the top spine. Default is True.
        hide_right (bool, optional): Whether to hide the right spine. Default is True.
    """
    if ax is None:
        ax = plt.gca()  # Get the current axis if no axis is provided

    # Toggle grid visibility
    ax.grid(grid)

    # Set the thickness of the left and bottom spines
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)

    # Hide the top and right spines if specified
    if hide_top:
        ax.spines['top'].set_visible(False)
    if hide_right:
        ax.spines['right'].set_visible(False)

    # Set font size for labels, titles, and tick labels
    ax.xaxis.label.set_size(fontsize)  # x-axis label font size
    ax.yaxis.label.set_size(fontsize)  # y-axis label font size
    ax.title.set_size(fontsize)        # Title font size
    ax.tick_params(axis='both', labelsize=fontsize)  # Tick label font size
    
    

# Add figure pannel
def add_fig_pannel(pannel_params, pannel="A"):
    x = pannel_params['x']
    y = pannel_params['y']
    fontsize = pannel_params['fontsize']
    fontweight = pannel_params['fontweight']
    plt.text(
        x,
        y,
        pannel,
        transform = plt.gca().transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        ha='right',
        va='top'
    )



# Function log_with_time
def printlg(message):
    """
    Display the current timestamp followed by the given message.

    Parameters:
        message (str): The message to log.

    Example:
        log_with_time("Processing started")
        # Output: [2024-12-23 14:30:45] Processing started
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")



def get_subfolders(target_folder):
    """
    Retrieves the names and absolute paths of subfolders within the specified folder 
    and returns them as a DataFrame.
    
    Parameters:
        target_folder (str): The path to the folder to be inspected.
    
    Returns:
        pd.DataFrame: A DataFrame containing the names and absolute paths of the subfolders.
    """
    # Retrieve the names and absolute paths of subfolders within the specified folder
    data = [{'folder_name': f.name, 'absolute_path': os.path.abspath(f.path)}
            for f in os.scandir(target_folder) if f.is_dir()]
    
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    
    return df



# Function to convert an array of bytes to an array of bits
def byte2bit(byte_array):
    """
    Convert an array of bytes to an array of bits.

    Parameters:
        byte_array (array-like): Input array containing bytes (0-255).

    Returns:
        np.ndarray: A 2D array of shape (N, 8) where each row represents
                    the 8 bits of the corresponding byte in the input.
    """
    
    # Type check
    if not isinstance(byte_array, (list, np.ndarray)):
        raise TypeError("Input must be a list or NumPy array.")

    # Convert input to NumPy array and flatten
    byte_array = np.asarray(byte_array, dtype=int).ravel()

    # Value range check
    if np.any(byte_array < 0) or np.any(byte_array > 255):
        raise ValueError("Input array must only contain 8-bit values (0-255).")

    # Create the shifts array for bit extraction
    shifts = np.arange(7, -1, -1)

    # Perform bit extraction and return result
    bit = ((byte_array[:, None] >> shifts) & 1).astype(int)
    return bit



# DataFrame Column Change Detection
def extract_transition_rows(df):
    """
    Extract rows where each column individually has changes.
    Adds row indices as a new column on the left and a column with 
    column names of changes as 'ColumnNameChanged' at the end.

    Parameters:
        df (pd.DataFrame): A DataFrame to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing rows where changes occur for each column.
                      Includes the row indices as the first column and a list of column
                      names with changes as the last column, sorted by RowIndex.

    Raises:
        ValueError: If the input is not a valid pandas DataFrame.
    """
    # Ensure the input is a valid pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty. Please provide a valid DataFrame.")

    try:
        # Calculate where changes occur for each column
        changes = df.ne(df.shift())  # True where values change

        # Create a DataFrame to store the results
        results = []

        for col_name in df.columns:
            # Find rows where this column changes
            col_changes = changes[col_name]
            rows_with_changes = df[col_changes].copy()
            rows_with_changes.insert(0, "Row_index", rows_with_changes.index)  # Add row indices
            rows_with_changes["Column_name_changed"] = col_name  # Add column name
            results.append(rows_with_changes)

        # Concatenate all results into a single DataFrame
        final_result = pd.concat(results, ignore_index=True)

        # Sort by RowIndex
        final_result = final_result.sort_values(by="Row_index").reset_index(drop=True)

        return final_result

    except Exception as e:
        # Generic error handling for unexpected issues
        raise RuntimeError(f"An error occurred while processing the DataFrame: {e}")
    
    


def round_custom_bins(x, bins):
    """
    Rounds a value based on specified bin ranges.

    Parameters:
        x (float or array-like): The value(s) to round.
        bins (list or array): A list of bin edges (e.g., [0, 2, 4, 6, 8]).

    Returns:
        float or array: The upper bound of the specified range.
    """
    bins = np.array(bins)  # NumPy 配列に変換
    index = np.searchsorted(bins, x, side='right') - 1  # 適切なビンを検索
    
    # `index + 1` が `bins` の範囲外にならないようにする
    index = np.clip(index + 1, 0, len(bins) - 1)
    
    return bins[index]  # 適切なビンの上限を返す

    



def delete_files(file_list):
    """
    Function to delete a specified list of files.

    Parameters:
        file_list (list): List of file paths to be deleted.

    Returns:
        dict: A dictionary indicating success (True) or failure (False) for each file.
        
    example:
    file_paths = [
            os.path.join(output_dir, "meta_data.pkl"),
            os.path.join(output_dir, "data.pkl"),
            os.path.join(output_dir, "fig.jpg")
        ]
    delete_files(file_paths)
    """
    results = {}

    for file_path in file_list:
        if os.path.exists(file_path):  # Check if the file exists
            os.remove(file_path)       # Delete the file
            print(f"Deleted: {file_path}")
            results[file_path] = True
        else:
            print(f"File not found: {file_path}")
            results[file_path] = False

    return results



def parse_time(time_str, date_str=None):
    """
    時間文字列を `datetime` オブジェクトに変換する。
    - `"%Y/%m/%d %H:%M:%S"` 形式ならそのまま変換
    - `"%Y/%m/%d %H:%M"` 形式なら秒を "00" に補完
    - `"%H:%M:%S"` 形式なら `date_str` から日付部分だけ取得して結合
    """
    formats = ["%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%H:%M:%S"]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            if fmt == "%H:%M:%S":  # "HH:MM:SS" の場合は `date_str` の日付部分を使用
                if date_str is None:
                    raise ValueError("Date string is required for time format '%H:%M:%S'.")
                # `date_str` から日付部分のみを取得
                date_obj = datetime.strptime(date_str.split()[0], "%Y/%m/%d")
                dt = datetime.combine(date_obj.date(), dt.time())  # 日付と時刻を結合
            return dt
        except ValueError:
            continue  # フォーマットが合わなければ次の形式を試す

    raise ValueError(f"Invalid date format: {time_str}")



def copy(
    source_path: str,
    output_path: str,
    path_mode: Literal['full', 'base', 'custom'] = 'full',
    custom_folder: Optional[str] = None,
    folders: Optional[List[str]] = None,
    extensions: Optional[List[str]] = None
):
    """
    Copy files or folders from source_path to output_path.

    Parameters:
    - path_mode: 
        'full'   : preserves full path structure (without drive letter)
        'base'   : uses only the base folder name
        'custom' : uses custom_folder name for destination
    - folders: List of subfolders to copy (copy all if None)
    - extensions: Only copy files with these extensions (e.g., ['.npy', '.mat'])
    """

    # Determine destination base path
    if path_mode == 'full':
        _, rel_path = os.path.splitdrive(source_path)
        rel_path = rel_path.lstrip("\\/")
        dest_base = os.path.join(output_path, rel_path)

    elif path_mode == 'base':
        dest_base = os.path.join(output_path, os.path.basename(source_path))

    elif path_mode == 'custom':
        if custom_folder is None:
            raise ValueError("custom_folder must be provided when path_mode is 'custom'")
        dest_base = os.path.join(output_path, custom_folder)

    else:
        raise ValueError("Invalid path_mode. Choose from 'full', 'base', or 'custom'.")

    # Define internal copy function
    def copy_with_filter(src_root, dst_root):
        for root, _, files in os.walk(src_root):
            for file in files:
                if extensions is None or os.path.splitext(file)[1] in extensions:
                    src_file = os.path.join(root, file)
                    rel_path = os.path.relpath(src_file, src_root)
                    dst_file = os.path.join(dst_root, rel_path)
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    shutil.copy2(src_file, dst_file)

    # Perform copy
    if folders is None:
        if extensions is None:
            shutil.copytree(source_path, dest_base, dirs_exist_ok=True)
            print(f"Copied entire directory → {dest_base}")
        else:
            copy_with_filter(source_path, dest_base)
            print(f"Copied filtered files to → {dest_base}")
    else:
        for folder in folders:
            src_folder = os.path.join(source_path, folder)
            dst_folder = os.path.join(dest_base, folder)

            if not os.path.exists(src_folder):
                print(f"Not found: {src_folder}")
                continue

            if extensions is None:
                shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
                print(f"Copied folder → {dst_folder}")
            else:
                copy_with_filter(src_folder, dst_folder)
                print(f"Copied filtered files from {folder} → {dst_folder}")

    return source_path, dest_base




def get_folder_size(path: str) -> int:
    """
    Recursively calculate the total size (in bytes) of all files in the given folder.
    """
    total_size = 0
    for root, _, files in os.walk(path):
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size



def bins_to_safe_filename(bins, precision=1):
    def float_to_str(x):
        s = f"{x:.{precision}f}".rstrip('0').rstrip('.')
        return s.replace('.', 'p')
    return 'bins_' + '_'.join(float_to_str(b) for b in bins)