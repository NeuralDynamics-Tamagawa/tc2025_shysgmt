import os
import numpy as np
import pandas as pd
from scipy import stats
import csv
import pickle
import libs.utils.utils as utils
import matplotlib.pyplot as plt


# Extract image list
def extract_img_list(input_dir):
    path = os.path.join(input_dir, "img")
    file_list = os.listdir(path)
    file_names = [os.path.splitext(file)[0] for file in file_list]
    numeric_values = []
    for file in file_names:
        try:
            numeric_values.append(int(file))  
        except ValueError:
            pass
    return numeric_values


# Replace out-of-list values with the previous valid value
def replace_single_anomalies(data):
    data = np.asarray(data)  
    if len(data) < 3:
        return data  #
    prev = np.roll(data, 1)  
    next_ = np.roll(data, -1)  
    anomaly_mask = (prev == next_) & (data != prev)
    data[anomaly_mask] = prev[anomaly_mask]
    return data


# Iteratively replace anomalies until no changes are made
def iterative_replace_anomalies(data):
    data = np.asarray(data)
    while True:
        new_data = replace_single_anomalies(data)
        if np.array_equal(new_data, data):  # 修正が行われなくなったら終了
            break
        data = new_data
    return data


# Extract transition rows
def extract_digits_from_string_array(string_array, start, end):
    # Convert the input to a NumPy array (as strings)
    arr = np.asarray(string_array, dtype=str)
    
    # Get the length of each string element (vectorized operation)
    lengths = np.char.str_len(arr)
    
    # Initialize the output array with float type to handle NaN values
    output = np.full((arr.shape[0], 1), np.nan)
    
    # Create a mask for rows where slicing is valid
    valid_mask = lengths >= end
    
    # Perform slicing only on rows where valid_mask is True
    # This uses a list comprehension but reduces processing as it operates only on valid rows
    valid_values = [int(float(x[start-1:end])) for x in arr[valid_mask]]
    
    # Assign extracted values to valid rows
    output[valid_mask, 0] = valid_values
    
    return output.flatten()




# Replace out-of-list values with the previous valid value
def replace_out_of_list_with_previous(data, valid_values):
    # Ensure the input data is converted to a NumPy array (optional if already ensured)
    data = np.asarray(data)
    
    # Convert valid_values to a set for faster lookup
    valid_values_set = set(valid_values)
    
    # Boolean mask indicating valid values (True: valid, False: invalid)
    mask = np.isin(data, list(valid_values_set))
    
    # Flatten the array (N×1 -> N): Forward fill operates in 1D
    arr = data.flatten()
    mask_flat = mask.flatten()

    # Assign indices to valid values; for invalid values, assign -1
    # Example: [False, True, True, False, True] -> [-1, 1, 2, -1, 4]
    idx_valid = np.where(mask_flat, np.arange(len(arr)), -1)

    # Propagate the indices of valid values forward using np.maximum.accumulate
    # This computes the "last valid value index" for each position
    np.maximum.accumulate(idx_valid, out=idx_valid)
    # Example: [-1, 1, 2, -1, 4] -> [-1, 1, 2, 2, 4]
    # Each position now holds the index of the last valid value encountered

    # Create a copy of the original array for output (to avoid modifying the input data)
    out = arr.copy()
    
    # For positions where idx_valid is -1, no valid value has appeared, so keep as is
    # For positions where idx_valid is not -1, replace with the value at the propagated index
    not_invalid_start = (idx_valid != -1)
    out[not_invalid_start] = arr[idx_valid[not_invalid_start]]

    return out



def photodiode_to_binary(photodiode, border = 30, window1=21, window2=31):
    # Convert photodiode data to binary
    temp = pd.Series(photodiode)
    # Smoothing data
    temp = temp.rolling(window=window1, center=True).mean()
    temp = temp.rolling(window=window2, center=True).mean()
    # Binaqrization
    binary_photodiode = np.where(temp > 30, 1, 0)
    utils.printlg("Done | Convert photodiode data to binary")
    return binary_photodiode








# ===================================================================
# ==================== dm2afc_gaborContrast_v001 ====================
# ===================================================================
def dm2afc_gaborContrast_v001(input_dir, output_dir, teensy, trial_start_state_num, reg_trial, pltshow=False):
   
    # Extracting digits from string array
    state      = extract_digits_from_string_array(teensy, 1, 3)
    stim       = extract_digits_from_string_array(teensy, 4, 6)
    teensy_io  = extract_digits_from_string_array(teensy, 10, 12)
    photodiode = extract_digits_from_string_array(teensy, 13, 15)
    cam1       = extract_digits_from_string_array(teensy, 16, 18)
    utils.printlg("Done | extract_digits_from_string_array")

    # Extracting valid values from stim
    img_list = extract_img_list(input_dir)

    # Replace out-of-list values with the previous valid value
    state      = replace_out_of_list_with_previous(state, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    stim       = replace_out_of_list_with_previous(stim, img_list)
    teensy_io  = replace_out_of_list_with_previous(teensy_io, np.arange(0, 256))
    photodiode = replace_out_of_list_with_previous(photodiode, np.arange(0, 256))
    cam1       = replace_out_of_list_with_previous(cam1, np.arange(0, 256))
    utils.printlg("Done | replace_out_of_list_with_previous")

    state = iterative_replace_anomalies(state)
    stim = iterative_replace_anomalies(stim)
    utils.printlg("Done | iterative_replace_anomalies")


    dat = [state, stim, teensy_io, photodiode, cam1]
    title = ["state", "stim", "teensy_io", "photodiode", "cam1"]
    save_path = os.path.join(output_dir, "hist_vals.jpg")
    plt.figure(figsize=(10, 10))
    for i_dat in range(len(dat)):
        histogram, bin_edges = np.histogram(dat[i_dat], bins=100, density=True)
        plt.subplot(3, 3, i_dat+1)
        plt.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor="black", align="edge")
        plt.xlabel('Value')
        plt.ylabel('Probability')
        plt.title(title[i_dat])
        if i_dat <= 1:
            plt.xticks(np.arange(1, max(dat[i_dat])+1, 1))  
    # Adjust the layout
    plt.tight_layout() 
    # Save the plot 
    plt.savefig(save_path, format="jpg", dpi=300)
    if bool(pltshow):
        plt.show()
    else:
        plt.close()
    utils.printlg("Done | Plot vals histogram")

    # Convert photodiode data to binary
    binary_photodiode = photodiode_to_binary(
        photodiode,
        border = 30,
        window1=21,
        window2=31
    )

    # Convert teensy_io to binary
    bit_array = utils.byte2bit(teensy_io)
    left_button  = bit_array[:, 0]
    right_button = bit_array[:, 1]
    sync         = bit_array[:, 3] 
    dispenser    = bit_array[:, 4]
    utils.printlg("Done | Convert teensy_io to binary")

    # Create a DataFrame from the processed data
    data_dict = {
    "State": state,
    "Stim": stim,
    "Left_button": left_button,
    "Right_button": right_button,
    "Dispenser": dispenser,
    "Binary_photodiode": binary_photodiode,
    "Sync": sync,
    "Photodiode": photodiode,
    "Cam1": cam1,
    }
    procTeensy = pd.DataFrame(data_dict, dtype='uint8')
    utils.printlg("Done | Generate df_procTeensy")


    # Extract transition rows
    event = utils.extract_transition_rows(procTeensy[["State", "Stim", "Left_button", "Right_button", "Dispenser", "Binary_photodiode"]])
    utils.printlg("Done | Extract transition rows")

    tssn = trial_start_state_num
    # Shift the 'State' column to get the previous row's value
    event['State_shifted'] = event['State'].shift(1, fill_value=0)
    event['new_trial'] = (event['State'] == tssn) & (event['State_shifted'] != tssn)
    event['Trial'] = event['new_trial'].cumsum()+reg_trial
    event = event.drop(columns=['State_shifted', 'new_trial'])
    utils.printlg("Done | Generate df_event")

     
    utils.printlg("End | Generate df_procTeensy and df_event")
    return procTeensy, event
        



# =========================================================================
# dm2afc_cursor_v001
# dm2afc_illusion_of_control_v001 
# =========================================================================
# Same function for both programs

def illusion_of_control(input_dir, output_dir, teensy, trial_start_state_num, reg_trial, pltshow=False):
    
    # Extracting digits from string array
    block      = extract_digits_from_string_array(teensy, 1, 3)
    state      = extract_digits_from_string_array(teensy, 4, 6)
    stim       = extract_digits_from_string_array(teensy, 7, 9)
    cursorval  = extract_digits_from_string_array(teensy, 10, 12)
    teensy_io  = extract_digits_from_string_array(teensy, 13, 15)
    photodiode = extract_digits_from_string_array(teensy, 16, 18)
    cam1       = extract_digits_from_string_array(teensy, 19, 21)
    cam2       = extract_digits_from_string_array(teensy, 22, 24)
    utils.printlg("Done | extract_digits_from_string_array")

    # Extracting valid values from stim
    img_list = extract_img_list(input_dir)
    
    # Replace out-of-list values with the previous valid value
    block      = replace_out_of_list_with_previous(block, [1, 2, 3])
    state      = replace_out_of_list_with_previous(state, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    stim       = replace_out_of_list_with_previous(stim, img_list)
    cursorval  = replace_out_of_list_with_previous(cursorval, np.arange(28, 229))
    teensy_io  = replace_out_of_list_with_previous(teensy_io, np.arange(0, 256))
    photodiode = replace_out_of_list_with_previous(photodiode, np.arange(0, 256))
    cam1       = replace_out_of_list_with_previous(cam1, np.arange(0, 256))
    cam2       = replace_out_of_list_with_previous(cam2, np.arange(0, 256))
    utils.printlg("Done | replace_out_of_list_with_previous")
    
    block = iterative_replace_anomalies(block)
    state = iterative_replace_anomalies(state)
    stim = iterative_replace_anomalies(stim)
    utils.printlg("Done | iterative_replace_anomalies")
    
    
    dat = [block, state, stim, cursorval, teensy_io, photodiode, cam1, cam2]
    title = ["block", "state", "stim", "cursorval", "teensy_io", "photodiode", "cam1", "cam2"]
    save_path = os.path.join(output_dir, "hist_vals.jpg")
    plt.figure(figsize=(10, 10))
    for i_dat in range(len(dat)):
        histogram, bin_edges = np.histogram(dat[i_dat], bins=100, density=True)
        plt.subplot(3, 3, i_dat+1)
        plt.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor="black", align="edge")
        plt.xlabel('Value')
        plt.ylabel('Probability')
        plt.title(title[i_dat])
        if i_dat <= 1:
            plt.xticks(np.arange(1, max(dat[i_dat])+1, 1))  
    # Adjust the layout
    plt.tight_layout() 
    # Save the plot 
    plt.savefig(save_path, format="jpg", dpi=300)
    if bool(pltshow):
        plt.show()
    else:
        plt.close()
    utils.printlg("Done | Plot vals histogram")
    
    # Convert photodiode data to binary
    binary_photodiode = photodiode_to_binary(
        photodiode,
        border = 30,
        window1=21,
        window2=31
    )
    
    # Convert teensy_io to binary
    bit_array = utils.byte2bit(teensy_io)
    left_button  = bit_array[:, 0]
    right_button = bit_array[:, 1]
    sync         = bit_array[:, 3] 
    dispenser    = bit_array[:, 4]
    utils.printlg("Done | Convert teensy_io to binary")
    
    # Create a DataFrame from the processed data
    data_dict = {
    "Block": block,
    "State": state,
    "Stim": stim,
    "Left_button": left_button,
    "Right_button": right_button,
    "Dispenser": dispenser,
    "Binary_photodiode": binary_photodiode,
    "Cursor_val": cursorval,
    "Sync": sync,
    "Photodiode": photodiode,
    "Cam1": cam1,
    "Cam2": cam2
    }
    procTeensy = pd.DataFrame(data_dict, dtype='uint8')
    utils.printlg("Done | Generate procTeensy")
    
    # Extract transition rows
    event = utils.extract_transition_rows(procTeensy[["Block", "State", "Stim", "Left_button", "Right_button", "Dispenser", "Binary_photodiode"]])
    utils.printlg("Done | Extract transition rows")
    
    tssn = trial_start_state_num
    # Shift the 'State' column to get the previous row's value
    event['State_shifted'] = event['State'].shift(1, fill_value=0)
    event['new_trial'] = (event['State'] == tssn) & (event['State_shifted'] != tssn)
    event['Trial'] = event['new_trial'].cumsum()+reg_trial
    event = event.drop(columns=['State_shifted', 'new_trial'])
    utils.printlg("Done | Generate event")
     
    utils.printlg("End | Generate df_procTeensy and df_event")
    return procTeensy, event


