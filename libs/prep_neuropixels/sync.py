import os
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import csv
import pickle
import libs.utils.utils as utils
import spikeinterface.extractors as se
import spikeinterface as si
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
from tqdm import tqdm


def generate_sync_imec(session_dir):

    utils.printlg('Gen. sync_imec | Session dir: '+session_dir)
    
    neuropixels_dir    = os.path.join(session_dir, 'neuropixels')
    processed_data_dir = os.path.join(session_dir, 'processed-data')

    neuropixels_subdirs     = utils.get_subfolders(neuropixels_dir)
    utils.printlg(neuropixels_subdirs)
    
    for i_imec in range(1, neuropixels_subdirs.shape[0]):

        # Generate folders
        imec_folder = 'imec'+str(i_imec)
        stream_name = 'imec'+str(i_imec)+'.ap'

        # Load dataset directories
        neuropixels_dir     = neuropixels_subdirs.iloc[i_imec]['absolute_path']
        utils.printlg('Gen. sync_imec | Loading data from ' + neuropixels_dir)
           

        # output directories
        save_dir = os.path.join(processed_data_dir, imec_folder)
        os.makedirs(save_dir, exist_ok=True)

        # Extract sync from spikeGLX
        recording = se.read_spikeglx(neuropixels_dir, stream_name='imec0.ap', load_sync_channel=True, )

        # Extract sync from spikeGLX
        utils.printlg('Gen. sync_imec | Saving sync channel')
        channel_ids = recording.get_channel_ids()
        sync_imec = recording.get_traces(channel_ids=[channel_ids[384]])
        sync_imec = sync_imec.ravel()
        
        file_name = imec_folder+'_'+channel_ids[384].replace('.','').replace('#','') + '.npy'
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, sync_imec)
        utils.printlg('Saved. | File path: ' + file_path)



def generate_sync_imec2(session_dir):
    utils.printlg('Gen. sync_imec | Session dir: ' + session_dir)

    neuropixels_dir = os.path.join(session_dir, 'neuropixels')
    processed_data_dir = os.path.join(session_dir, 'processed-data')

    neuropixels_subdirs = utils.get_subfolders(neuropixels_dir)

    for i_imec in range(neuropixels_subdirs.shape[0]):

        start_time = time.time()

        imec_folder = 'imec' + str(i_imec)
        stream_name = 'imec' + str(i_imec) + '.ap'

        neuropixels_dir = neuropixels_subdirs.iloc[i_imec]['absolute_path']

        save_dir = os.path.join(processed_data_dir, imec_folder)
        os.makedirs(save_dir, exist_ok=True)

        utils.printlg(f'Gen. sync_imec | Loading data from {neuropixels_dir}')
        recording = se.read_spikeglx(neuropixels_dir, stream_name='imec0.ap', load_sync_channel=True)

        load_time = time.time()
        utils.printlg(f'Data loading started...')

        channel_ids = recording.get_channel_ids()
        num_samples = recording.get_num_frames()
        chunk_size = 50000  # 一度に読み込むサンプル数
        sync_imec = []

        with tqdm(total=num_samples, desc=f'Loading sync channel {channel_ids[384]}', unit=' samples') as pbar:
            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)
                chunk = recording.get_traces(start_frame=start, end_frame=end, channel_ids=[channel_ids[384]])
                sync_imec.append(chunk.ravel())  # Convert to 1D array
                pbar.update(chunk.shape[1])  # Update progress
        
        sync_imec = np.concatenate(sync_imec)  # Concatenate chunks
        utils.printlg(f'Data loaded in {time.time() - load_time:.2f} seconds')

        # Saving
        file_name = imec_folder + '_' + channel_ids[384].replace('.', '').replace('#', '') + '.npy'
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, sync_imec)

        # Recoding processing time
        save_time = time.time()
        utils.printlg(f'Saved. | File path: {file_path}')
        utils.printlg(f'Total processing time: {save_time - start_time:.2f} seconds (Load: {load_time - start_time:.2f}s, Save: {save_time - load_time:.2f}s)')
    
    
    
        
'''
Generate procTeensy_to_imec mapping array
- Removing noise from procTeensy sync data
- Plot confirmation of synchronization
- load procTeensy.pkl file
- Retrieve 'apSY0.npy' file from imec_dir
- Compare synchronization signals from procTeensy and imec and adjust for lag
'''


def remove_noise(signal):
    signal = np.array(signal)
    
    original_signal = signal.copy()
    
    d2_signal =  np.diff(np.diff(signal))
    change_points = np.where(abs(d2_signal)==2)[0]+1

    filled_signal = signal.copy()

    if len(change_points) == 0:
        return filled_signal  # Return original signal if no change points exist

    # Get previous and next values using np.roll() for efficient vectorized computation
    prev_values = np.roll(filled_signal, shift=1)  # Shift left
    next_values = np.roll(filled_signal, shift=-1) # Shift right

    # Handle boundary cases (first and last elements)
    prev_values[0] = filled_signal[1]    # First element correction
    next_values[-1] = filled_signal[-2]  # Last element correction

    # Interpolate change points using the average of previous and next values
    filled_signal[change_points] = (prev_values[change_points] + next_values[change_points]) / 2
    
    return filled_signal


        
def plot_sync_confirm(edge_idx_imec_filtered, edge_idx_procTeensy, figure_dir, n_imec):

    reg = 6
    row = 2
    col = 5
    dpT = np.diff(edge_idx_procTeensy.astype(float))
    dimec = np.diff(edge_idx_imec_filtered.astype(float))
    residual = (dpT*reg - dimec)

    perc25_idx = int(len(residual)*0.25)
    perc50_idx = int(len(residual)*0.50)
    perc75_idx = int(len(residual)*0.75)

    fig = plt.figure(figsize=(col*6, row*5))
    gs = fig.add_gridspec(row, col)  

    ax = fig.add_subplot(gs[0, 0:5])
    plt.plot(residual)
    plt.title("Residual")
    plt.ylim(-50, 50)

    ax = fig.add_subplot(gs[1, 0])
    plt.plot(np.diff(edge_idx_imec_filtered[:50]/reg))
    plt.plot(np.diff(edge_idx_procTeensy[:50]))
    plt.title("Start")

    ax = fig.add_subplot(gs[1, 1])
    plt.plot(np.diff(edge_idx_imec_filtered[perc25_idx-25:perc25_idx+25]/reg))
    plt.plot(np.diff(edge_idx_procTeensy[perc25_idx-25:perc25_idx+25]))
    plt.title("25%")

    ax = fig.add_subplot(gs[1, 2])
    plt.plot(np.diff(edge_idx_imec_filtered[perc50_idx-25:perc50_idx+25]/reg))
    plt.plot(np.diff(edge_idx_procTeensy[perc50_idx-25:perc50_idx+25]))
    plt.title("50%")

    ax = fig.add_subplot(gs[1, 3])
    plt.plot(np.diff(edge_idx_imec_filtered[perc75_idx-25:perc75_idx+25]/reg))
    plt.plot(np.diff(edge_idx_procTeensy[perc75_idx-25:perc75_idx+25]))
    plt.title("75%")

    ax = fig.add_subplot(gs[1, 4])
    plt.plot(np.diff(edge_idx_imec_filtered[-50:])/reg)
    plt.plot(np.diff(edge_idx_procTeensy[-50:]))
    plt.title("End")


    plt.suptitle("Confirmation synchronization", fontsize=14)
    file_name = 'sync_confirmation_imec'+str(n_imec)+'.png'
    plt.savefig(os.path.join(figure_dir, file_name))
    #plt.show()
    #plt.close()
        


def load_procTeensy(procTeensy_dir):
    """Load procTeensy.pkl file"""
    file_path = os.path.join(procTeensy_dir, 'procTeensy.pkl')

    if not os.path.exists(file_path):
        utils.printlg(f"Error: procTeensy.pkl not found in {procTeensy_dir}. Skipping execution.")
        return None

    try:
        with open(file_path, 'rb') as f:
            procTeensy = pickle.load(f)
        sync_procTeensy = procTeensy.get('Sync')

        if sync_procTeensy is None:
            utils.printlg("Error: 'Sync' key not found in procTeensy.pkl. Skipping execution.")
            return None

        return sync_procTeensy

    except Exception as e:
        utils.printlg(f"Error loading procTeensy.pkl: {e}")
        return None



def get_imec_sync_file(imec_dir):
    """Retrieve 'apSY0.npy' file from imec_dir"""
    imec_sync_files = [f for f in os.listdir(imec_dir) if f.endswith('apSY0.npy')]

    if not imec_sync_files:
        utils.printlg(f"Error: No 'apSY0.npy' file found in {imec_dir}. Skipping this iteration.")
        return None

    return os.path.join(imec_dir, imec_sync_files[0])



def process_sync_signals(sync_procTeensy, sync_imec):
    """Compare synchronization signals from procTeensy and imec and adjust for lag"""
    edge_idx_imec = np.where(np.diff(sync_imec) != 0)[0]  
    edge_idx_procTeensy = np.where(np.diff(sync_procTeensy) != 0)[0] 

    # Set a valid range to avoid out-of-bounds errors
    start, end = 0, min(10000, len(edge_idx_imec), len(edge_idx_procTeensy))  
    corr = np.correlate(np.diff(edge_idx_imec[start:end]), np.diff(edge_idx_procTeensy[start:end]), mode="full")
    lags = np.arange(-len(edge_idx_imec[start:end]) + 1, len(edge_idx_procTeensy[start:end]))
    
    lag = lags[np.argmax(corr)]
    edge_idx_imec_filtered = edge_idx_imec[max(0, lag+1):][:len(edge_idx_procTeensy)]

    return edge_idx_procTeensy, edge_idx_imec_filtered



# Main function (Generate procTeensy_to_imec mapping)
def generate_procTeensy_to_imec(session_dir):
    """Align procTeensy synchronization data with imec synchronization data"""

    # Set directories
    processed_data_dir = os.path.join(session_dir, 'processed-data')
    procTeensy_dir     = os.path.join(session_dir, 'procTeensy')
    figure_dir         = os.path.join(session_dir, 'figure', 'sync')
    
    os.makedirs(figure_dir, exist_ok=True)
    
    # Retrieve all subdirectories under 'processed-data'
    processed_data_subdirs = utils.get_subfolders(processed_data_dir)

    # Load procTeensy synchronization data
    sync_procTeensy = load_procTeensy(procTeensy_dir)
    if sync_procTeensy is None:
        return  # Terminate execution if an error occurs

    # Remove noise from procTeensy sync data
    sync_procTeensy = remove_noise(sync_procTeensy)

    # Iterate over each imec directory
    for i_imec, subdir in processed_data_subdirs.iterrows():
        utils.printlg(f'Generating procTeensy_to_imec | i_imec={i_imec}')

        # Get imec synchronization file
        imec_dir = subdir["absolute_path"]
        imec_file_path = get_imec_sync_file(imec_dir)

        if imec_file_path is None:
            continue  # Skip iteration if no valid file is found

        # Load imec sync data
        sync_imec = np.load(imec_file_path)

        # Process synchronization signals and adjust for lag
        edge_idx_procTeensy, edge_idx_imec_filtered = process_sync_signals(sync_procTeensy, sync_imec)

        # Generate confirmation plot
        plot_sync_confirm(edge_idx_imec_filtered, edge_idx_procTeensy, figure_dir, n_imec=i_imec)

        # Create procTeensy_to_imec mapping
        procTeensy_to_imec = np.zeros(np.max(edge_idx_procTeensy) + 1)

        for i in range(len(edge_idx_procTeensy) - 1):
            row_pt = np.arange(edge_idx_procTeensy[i], edge_idx_procTeensy[i+1])  
            row_imec = np.linspace(edge_idx_imec_filtered[i], edge_idx_imec_filtered[i+1] - 6, len(row_pt))  
            procTeensy_to_imec[row_pt.astype(int)] = row_imec  

        # Normalize by sampling rate
        procTeensy_to_imec /= 30000  

        # Save the processed data
        save_path = os.path.join(imec_dir, f'procTeensy_to_imec{i_imec}.npy')
        print(procTeensy_to_imec)
        np.save(save_path, procTeensy_to_imec)
        utils.printlg(f'Saved: {save_path}')






if __name__ == "__main__":
    file_path = r'Z:\Handle_data\mainDirs_for_IoC.csv'
    IoC_dir_Info = pd.read_csv(file_path)
    utils.printlg(IoC_dir_Info)

    for i_session in range(5, len(IoC_dir_Info)):
        session_dir = IoC_dir_Info.iloc[i_session]['Session_dir']
        utils.printlg('Session dir: '+session_dir)
        generate_procTeensy_to_imec(session_dir)
        utils.printlg('Done')