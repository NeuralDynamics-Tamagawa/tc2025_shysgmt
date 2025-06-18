import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pickle
import libs.utils.utils as utils
import libs.handle.dataset as handle_dataset
import libs.analysis.behavior_single_session as ana_behv_sglssn



# Generate trials from the event data
def generate_trials_auto(root_dir, target_subjects):
    
    subject_folders = utils.get_subfolders(root_dir)
    subject_folders = subject_folders[subject_folders['folder_name'].isin(target_subjects)]
    subject_folders['sort_key'] = subject_folders['folder_name'].apply(lambda x: target_subjects.index(x))
    subject_folders = subject_folders.sort_values('sort_key').reset_index(drop=True)

    for i_subject in range(0, len(subject_folders)):
        session_folders = utils.get_subfolders(subject_folders.iloc[i_subject] ['absolute_path'])

        for i_session in range(0, len(session_folders)):
            session_dir = session_folders.loc[i_session]['absolute_path']
            generate_trials_session(session_dir, save_dir=None)
                
                



# Local functions (validity check)
def count_valid_transitions(df, state_transitions):
    validation = np.zeros(len(state_transitions), dtype=int)  
    for i, (prev_states, current_states) in enumerate(state_transitions):
        idx = (df['State'].isin(current_states)) & (df['State'].shift(1).isin(prev_states))
        validation[i] = idx.sum()  
    return validation


def extract_metadata(meta_file):
    header       = meta_file["Header"]
    subject      = header.loc[header["Name"]=="subject", "Value"].iloc[0]
    program_name = header.loc[header["Name"] == "programName", "Value"].iloc[0]
    descriptions  = header.loc[header["Name"]=="descriptions", "Value"].iloc[0]
    time         = meta_file["Time"]
    
    datetime     = time.loc[time["Name"]=="Datetime", "Value"].iloc[0]
    start_time   = time.loc[time["Name"]=="Start", "Value"].iloc[0]
    end_time     = time.loc[time["Name"]=="End", "Value"].iloc[0]
    start_time   = utils.parse_time(time_str=start_time, date_str=datetime)
    end_time     = utils.parse_time(time_str=end_time, date_str=datetime)
    
    return subject, program_name, descriptions, start_time, end_time




def generate_trials(meta_file, event):
    
    header = meta_file['Header']
    program_name = header.loc[header["Name"] == "programName", "Value"] 

    if program_name.values[0] == 'dm2afc_gc_v001':
        trials = cursor(meta_file, event)
    elif program_name.values[0] == 'dm2afc_cousor_v001':
        trials = cursor(meta_file, event)
    elif program_name.values[0] == 'dm2afc_cursor_v001':
        trials = cursor(meta_file, event)
    elif program_name.values[0] == 'dm2afc_illusion_of_control_v001':
        trials = illusion_of_control(meta_file, event)
    elif program_name.values[0] == 'dm2afc_illusion_of_control_v002':
        trials = illusion_of_control(meta_file, event)
    else:
        print("Error")
    
    return trials




def generate_trials_session(session_dir, save_dir=None):
    input_dir = os.path.join(session_dir, 'procTeensy')
    if save_dir is None:
        save_dir = input_dir
    
    files_to_check = ['metaFile.pkl', 'event.pkl']
    existing_files = [file for file in files_to_check if os.path.exists(os.path.join(input_dir, file))]
    
    if len(existing_files) == len(files_to_check):
        
        # Loading
        session   = handle_dataset.Session(session_dir)
        meta_file = session.metaFile
        event     = handle_dataset.Session.load_event(session)
            
        # Processing
        trials = generate_trials(meta_file, event)
        trials.to_pickle(os.path.join(save_dir, "trials.pkl"))
        utils.printlg("Session dir: " + session_dir)
        return trials
    else:
        print("Error | Missing files")
        return None
    


def load_pooled_trials(root_dir, target_subjects, save_path=None, check=False):
    if root_dir == None:
        root_dir = 'Z:/Data'
    subject_folders = utils.get_subfolders(root_dir)
    subject_folders = subject_folders[subject_folders['folder_name'].isin(target_subjects)]
    subject_folders['folder_name'] = pd.Categorical(subject_folders['folder_name'], categories=target_subjects, ordered=True)
    subject_folders = subject_folders.sort_values('folder_name')

    grouped_trials = []

    for i_subject in range(0, len(subject_folders)):
        session_folders = utils.get_subfolders(subject_folders.iloc[i_subject] ['absolute_path'])
        session_folders = session_folders.sort_values('folder_name').reset_index(drop=True)
        utils.printlg(subject_folders.iloc[i_subject]["folder_name"])
        
        for i_session in range(0, len(session_folders)):

                session_folder = session_folders.loc[i_session]['absolute_path']
                if check:
                    print("    ["+str(i_session)+"]"+session_folder)
                
                session = handle_dataset.Session(session_folder)
                
                # Loading session data
                trials    = session.trials
                
                # Filtering trials
                df = trials.copy(deep=True)
                df["Session_no"] = i_session+1
            
                df = df.dropna(subset=['Trial'])
                
                # Grouping
                grouped_trials.append(df)
        
    pool_trials = pd.concat(grouped_trials, ignore_index=True)
    
    if save_path is not None:
        pool_trials.to_pickle(save_path)
    
    return pool_trials



def augment_trials_pooled(trials_pooled, subjects):
    
    df = trials_pooled.copy()
    
    df = df[df['Subject'].isin(subjects)]

    correct = df['Correct'].map({True: 1, False: -1}).astype(int) 
    contrast_abs = df['Contrast'].abs()

    df['Reaction_time'] = np.where(df['Choice'] == 'left', df['RT_left'], df['RT_right'])
    df['Contrast_abs'] = df['Contrast'].abs()
    df['pRight'] = df['Choice'].map({'left': 0, 'right': 1}).astype(int)
    df['Pushsum'] = df['PC_left']+df['PC_right']
    df['Initpushsum'] = df['IPC_left']+df['IPC_right']
    df['Pre_contrast'] = df['Contrast'].shift(-1)
    df['Pre_choice'] = df['Choice'].shift(-1)
    df['Pre_initpushsum'] = df['IPC_left'].shift(-1)+df['IPC_right'].shift(-1)
    df['Pre_correct'] = df['Correct'].shift(-1)
    df['Pre_evidence'] = correct.shift(-1) * contrast_abs.shift(-1)

    first_trial = df['Start_datetime'] != df['Start_datetime'].shift()
    last_trial = df['Start_datetime'] != df['Start_datetime'].shift(-1)
    TF = first_trial | last_trial
    df = df[~TF]

    trials_pooled = df.copy()
    
    return trials_pooled
    




# Generate trials for cursor task
def cursor(meta_file, event):
    contrast_list = meta_file["Contrast"]
    contrast_list['Low_contrast'] = pd.to_numeric(contrast_list['Low_contrast'], errors='coerce')
    contrast_list['High_contrast'] = pd.to_numeric(contrast_list['High_contrast'], errors='coerce')


    event_grouped = event.groupby('Trial')

    sampling_rate = 5000  # Sampling frequency
    num_trials = len(event_grouped.groups) -1 
    
    block           = [np.nan] * num_trials
    n_trial         = [np.nan] * num_trials
    stim            = [np.nan] * num_trials
    cue_left        = [np.nan] * num_trials
    cue_right       = [np.nan] * num_trials
    contrast        = [np.nan] * num_trials
    choice          = [np.nan] * num_trials
    correct         = np.zeros(num_trials, dtype=bool)
    delay           = np.zeros(num_trials, dtype=bool)
    reward          = np.zeros(num_trials, dtype=bool)
    omission        = np.zeros(num_trials, dtype=bool)
    stim_delay      = [np.nan] * num_trials
    reward_delay    = [np.nan] * num_trials
    RT_left         = [np.nan] * num_trials
    RT_right        = [np.nan] * num_trials
    selection_time  = [np.nan] * num_trials
    time_investment = [np.nan] * num_trials
    PC_left         = [np.nan] * num_trials
    PC_right        = [np.nan] * num_trials
    IPC_left        = [np.nan] * num_trials
    IPC_right       = [np.nan] * num_trials


    for i_trial in range(1, num_trials):

        df = event_grouped.get_group(i_trial)
        
        state_transitions = [
            ([1], [2]),          # 2 → 1
            ([2], [3, 4]),       # 3 or 4 → 2
            ([3, 4], [5, 6, 7, 8]),  # 5, 6, 7, 8 → 3 or 4
            ([5, 6, 7, 8], [9, 10, 11, 12]), # 9 or 10 → 5, 6, 7, 8
        ]

        validation = count_valid_transitions(df, state_transitions)
        
        if validation.sum() != len(state_transitions):
            continue
            # Check if the trial is valid
    
        state = df['State'].values
        stim_code = df.loc[(df["State"] == 3) | (df["State"] == 4), "Stim"]
        stim_code = np.unique(stim_code)
        stim_code = stim_code[stim_code >= 10]
        mod = stim_code % 10

        # Trial number
        n_trial[i_trial] = i_trial


        # Stim left or right
        if np.any(state == 3):
            stim[i_trial] = "left"
        elif np.any(state == 4):
            stim[i_trial] = "right"
        else:
            stim[i_trial] = np.nan


        # Cue_left Cue_right
        if stim_code.size == 0:
            contrast[i_trial] = np.nan
        elif (stim_code >= 10) & (stim_code < 20):
            cue_left[i_trial] = contrast_list['High_contrast'][mod].values[0]
            cue_right[i_trial] = contrast_list['Low_contrast'][mod].values[0]
        elif (stim_code >= 20) & (stim_code < 30):
            cue_left[i_trial] = contrast_list['Low_contrast'][mod].values[0]
            cue_right[i_trial] = contrast_list['High_contrast'][mod].values[0]
        else :
            contrast[i_trial] = np.nan


        # Contrast
        if stim_code.size == 0:
            contrast[i_trial] = np.nan
        elif (stim_code >= 10) & (stim_code < 20):
            temp = contrast_list['Low_contrast'][mod] - contrast_list['High_contrast'][mod]
            contrast[i_trial] = temp.values[0] if temp.size > 0 else np.nan
        elif (stim_code >= 20) & (stim_code < 30):
            temp = contrast_list['High_contrast'][mod] - contrast_list['Low_contrast'][mod]
            contrast[i_trial] = temp.values[0] if temp.size > 0 else np.nan
        else :
            contrast[i_trial] = np.nan


        # Choice
        if np.any(state == 5):
            choice[i_trial] = "left"
        elif np.any(state == 6):
            choice[i_trial] = "right"
        elif np.any(state == 7):
            choice[i_trial] = "right"
        elif np.any(state == 8):
            choice[i_trial] = "left"
        else:
            choice[i_trial] = np.nan
            
            
        # Correct
        if np.any(state == 3) & np.any(state == 5):
            correct[i_trial] = True
        elif np.any(state == 4) & np.any(state == 6):
            correct[i_trial] = True
        elif np.any(state == 3) & np.any(state == 7):
            correct[i_trial] = False
        elif np.any(state == 4) & np.any(state == 8):
            correct[i_trial] = False
        else:
            correct[i_trial] = False


        # Delay
        if np.any((state == 9) | np.any(state == 10)):
            delay[i_trial] = True
        else: 
            delay[i_trial] = False

        # Reward
        if np.any((state == 11)):
            reward[i_trial] = True
        else:
            reward[i_trial] = False  
                
        # Omission
        if np.any((state == 12)): 
            omission[i_trial] = True
        else:
            omission[i_trial] = False
        

        if i_trial > 0 & i_trial < num_trials-1:
        
            # Selection time
            # Stim onset
            tf = (df['State'].isin([3, 4])) & (df['State'].shift(1).isin([2]))
            row_idx_stim = df.loc[tf, 'Row_index'].values[0]
            
            # Selection onset
            tf = (df['State'].isin([5, 6, 7, 8])) & (df['State'].shift(1).isin([3, 4]))
            row_idx_select = df.loc[tf, 'Row_index'].values[0]
            
            # Reward offset 
            subset = df[df['Column_name_changed'] == 'Dispenser'].copy()
            matched = subset.loc[subset['Row_index'].diff() == 500, 'Row_index']
            if not matched.empty:
                row_idx_dispenser = matched.values[0]
                reward_delay[i_trial] = (row_idx_dispenser - row_idx_select) / sampling_rate
            else:
                reward_delay[i_trial] = None  # Set to None if no matching rows are found
            
            # --- Detect photodiode-based stimulus onset during State 3 or 4 ---
            tf = (df['Column_name_changed'] == 'Binary_photodiode') & (df['State'].isin([3, 4]))
            matched_rows = df.loc[tf, 'Row_index']

            # Initialize as NaN by default
            row_idx_stim_photodiode = row_idx_stim
            stim_delay[i_trial] = np.nan

            # Proceed only if a matching row exists
            if not matched_rows.empty:
                candidate_idx = matched_rows.values[0]
                
                # Accept only if photodiode onset is at or after stim timing
                if candidate_idx >= row_idx_stim:
                    row_idx_stim_photodiode = candidate_idx
                    stim_delay[i_trial] = (row_idx_stim_photodiode - row_idx_stim) / sampling_rate
            
            # Time calculations
            # Selection time
            selection_time[i_trial] = (row_idx_select - row_idx_stim_photodiode) / sampling_rate


            # Reaction time
            # RT Left
            tf = (df['Column_name_changed'].isin(['Left_button'])) & (df['Left_button'].isin([1])) & (df['State'].isin([3, 4]))
            if not df.loc[tf, 'Row_index'].empty:
                row_idx_RT_L = df.loc[tf, 'Row_index'].values[0]
                RT_left[i_trial] = (row_idx_RT_L - row_idx_stim_photodiode) / sampling_rate
            else:
                RT_left[i_trial] = None  # Set to None if no matching rows are found
                
            # RT Light
            tf = (df['Column_name_changed'].isin(['Right_button'])) & (df['Right_button'].isin([1])) & (df['State'].isin([3, 4]))
            if not df.loc[tf, 'Row_index'].empty:
                row_idx_RT_R = df.loc[tf, 'Row_index'].values[0]
                RT_right[i_trial] = (row_idx_RT_R - row_idx_stim_photodiode) / sampling_rate
            else:
                RT_right[i_trial] = None  # Set to None if no matching rows are found
                
                
            # Push count left and right (Choice)
            df_push_count = df.loc[(df["State"] == 3) | (df["State"] == 4)]

            cnd1 = df_push_count["Column_name_changed"] == "Left_button"
            cnd2 = df_push_count["Left_button"] == 1
            PC_left[i_trial] = len(df_push_count.loc[cnd1 & cnd2])

            cnd1 = df_push_count["Column_name_changed"] == "Right_button"
            cnd2 = df_push_count["Right_button"] == 1
            PC_right[i_trial] = len(df_push_count.loc[cnd1 & cnd2])
            
            
    # Session meta data
    subject, program_name, descriptions, start_time, end_time = extract_metadata(meta_file)
    
    # Status
    status = np.full_like(choice, 'Other', dtype=object)
    status[np.logical_not(correct) & np.logical_not(reward)] = 'Error'
    status[np.logical_and(correct, reward)] = 'Correct'
    
    # Create DataFrame
    trials = pd.DataFrame({ 'Subject': subject,
                            'Program_name': program_name,
                            'Start_datetime': start_time,
                            'End_datetime': end_time,
                            'Descriptions': descriptions,
                            'Block': block,
                            'Trial': n_trial, 
                            'Stim': stim, 
                            'Cue_left': cue_left, 
                            'Cue_right': cue_right, 
                            'Contrast': contrast, 
                            'Choice': choice, 
                            'Correct': correct,
                            'Delay': delay,
                            'Reward': reward,
                            'Omission': omission,
                            'Status': status,
                            'Stim_delay': stim_delay,
                            'Reward_delay': reward_delay,
                            'RT_left': RT_left,
                            'RT_right': RT_right,
                            'Selection_time': selection_time,
                            'Time_investment': time_investment,
                            'PC_left': PC_left,
                            'PC_right': PC_right,
                            'IPC_left': IPC_left,
                            'IPC_right': IPC_right,
                            })
    
    # Add conditional probabilities
    cnd_prob_prestim = ana_behv_sglssn.conditional_probabilities_prestim(trials, alter_index_border=0.8)
    trials["Debias"] = cnd_prob_prestim["alter_counts_index"]
    
    # Add recording 
    tf_description = "rec" in descriptions.strip()
    trials["Recording"] = int(tf_description)
    
    # Contrast list
    df = trials.copy(deep=True)
    df = df.dropna(subset=['Trial'])
    trials["Num_diff_contrast"] = np.abs(df['Contrast']).unique().shape[0] 
    
    # Chronometric function
    trials["Chronofunc_pvalue"] = 1
    
    
    return trials



# Generate trials for illusion of control task (Behavioral confidence reporting task) 
def illusion_of_control(meta_file, event):
    
    contrast_list = meta_file["Contrast"]
    contrast_list['Low_contrast'] = pd.to_numeric(contrast_list['Low_contrast'], errors='coerce')
    contrast_list['High_contrast'] = pd.to_numeric(contrast_list['High_contrast'], errors='coerce')


    event_grouped = event.groupby('Trial')

    sampling_rate = 5000  # Sampling frequency
    num_trials = len(event_grouped.groups) -1 
    
    block           = [np.nan] * num_trials
    n_trial         = [np.nan] * num_trials
    stim            = [np.nan] * num_trials
    cue_left        = [np.nan] * num_trials
    cue_right       = [np.nan] * num_trials
    contrast        = [np.nan] * num_trials
    choice          = [np.nan] * num_trials
    correct         = np.zeros(num_trials, dtype=bool)
    delay           = np.zeros(num_trials, dtype=bool)
    reward          = np.zeros(num_trials, dtype=bool)
    omission        = np.zeros(num_trials, dtype=bool)
    stim_delay      = [np.nan] * num_trials
    reward_delay    = [np.nan] * num_trials
    RT_left         = [np.nan] * num_trials
    RT_right        = [np.nan] * num_trials
    selection_time  = [np.nan] * num_trials
    time_investment = [np.nan] * num_trials
    PC_left         = [np.nan] * num_trials
    PC_right        = [np.nan] * num_trials
    IPC_left        = [np.nan] * num_trials
    IPC_right       = [np.nan] * num_trials


    for i_trial in range(1, num_trials):


        df = event_grouped.get_group(i_trial)
        df_next = event_grouped.get_group(i_trial+1)
        
        state_transitions = [
            ([2], [3, 4]),       # 2  ->  3 or 4
            ([3, 4], [5, 6, 7, 8]),  # 3 or 4  ->  5, 6, 7, 8
            ([5, 6, 7, 8], [9, 10, 13, 14]), # 5, 6, 7, 8 => 9, 10, 13, 14
        ]

        validation = count_valid_transitions(df, state_transitions)
        
        if validation.sum() != 3:
            continue
            # Check if the trial is valid
        
        state = df['State'].values
        stim_code = df.loc[(df["State"] == 3) | (df["State"] == 4), "Stim"]
        stim_code = np.unique(stim_code)
        stim_code = stim_code[stim_code >= 10]
        mod = stim_code % 10

        # Block
        b = df.loc[(df["State"] == 3) | (df["State"] == 4), "Block"]
        block[i_trial] = np.array(b.unique()[0])
        
        # Trial number
        n_trial[i_trial] = i_trial


        # Stim left or right
        if np.any(state == 3):
            stim[i_trial] = "left"
        elif np.any(state == 4):
            stim[i_trial] = "right"
        else:
            stim[i_trial] = np.nan


        # Cue_left Cue_right
        if stim_code.size == 0:
            contrast[i_trial] = np.nan
        elif (stim_code >= 10) & (stim_code < 20):
            cue_left[i_trial] = contrast_list['High_contrast'][mod].values[0]
            cue_right[i_trial] = contrast_list['Low_contrast'][mod].values[0]
        elif (stim_code >= 20) & (stim_code < 30):
            cue_left[i_trial] = contrast_list['Low_contrast'][mod].values[0]
            cue_right[i_trial] = contrast_list['High_contrast'][mod].values[0]
        else :
            contrast[i_trial] = np.nan


        # Contrast
        if stim_code.size == 0:
            contrast[i_trial] = np.nan
        elif (stim_code >= 10) & (stim_code < 20):
            temp = contrast_list['Low_contrast'][mod] - contrast_list['High_contrast'][mod]
            contrast[i_trial] = temp.values[0] if temp.size > 0 else np.nan
        elif (stim_code >= 20) & (stim_code < 30):
            temp = contrast_list['High_contrast'][mod] - contrast_list['Low_contrast'][mod]
            contrast[i_trial] = temp.values[0] if temp.size > 0 else np.nan
        else :
            contrast[i_trial] = np.nan


        # Choice
        if np.any(state == 5):
            choice[i_trial] = "left"
        elif np.any(state == 6):
            choice[i_trial] = "right"
        elif np.any(state == 7):
            choice[i_trial] = "right"
        elif np.any(state == 8):
            choice[i_trial] = "left"
        else:
            choice[i_trial] = np.nan
            
            
        # Correct
        if np.any(state == 3) & np.any(state == 5):
            correct[i_trial] = True
        elif np.any(state == 4) & np.any(state == 6):
            correct[i_trial] = True
        elif np.any(state == 3) & np.any(state == 7):
            correct[i_trial] = False
        elif np.any(state == 4) & np.any(state == 8):
            correct[i_trial] = False
        else:
            correct[i_trial] = False


        # Delay
        if np.any((state == 9) | np.any(state == 10)):
            delay[i_trial] = True
        else: 
            delay[i_trial] = False

        # Reward
        if np.any((state == 11) | np.any(state == 12)):
            reward[i_trial] = True
        else:
            reward[i_trial] = False  
                
        # Omission
        if np.any((state == 13) | (state == 14)): 
            omission[i_trial] = True
        else:
            omission[i_trial] = False



        if i_trial > 0 & i_trial < num_trials-1:
        
            # Selection time / Time investment
            # Stim onset
            tf = (df['State'].isin([3, 4])) & (df['State'].shift(1).isin([2]))
            row_idx_stim = df.loc[tf, 'Row_index'].values[0]
            
            # --- Detect photodiode-based stimulus onset during State 3 or 4 ---
            tf = (df['Column_name_changed'] == 'Binary_photodiode') & (df['State'].isin([3, 4]))
            matched_rows = df.loc[tf, 'Row_index']

            # Initialize as NaN by default
            row_idx_stim_photodiode = row_idx_stim
            stim_delay[i_trial] = np.nan

            # Proceed only if a matching row exists
            if not matched_rows.empty:
                candidate_idx = matched_rows.values[0]
                
                # Accept only if photodiode onset is at or after stim timing
                if candidate_idx >= row_idx_stim:
                    row_idx_stim_photodiode = candidate_idx
                    stim_delay[i_trial] = (row_idx_stim_photodiode - row_idx_stim) / sampling_rate
                    
                
            # Selection onset
            tf = (df['State'].isin([5, 6, 7, 8])) & (df['State'].shift(1).isin([3, 4]))
            row_idx_select = df.loc[tf, 'Row_index'].values[0]
            
            # Reward offset 
            subset = df[df['Column_name_changed'] == 'Dispenser'].copy()
            matched = subset.loc[subset['Row_index'].diff() == 500, 'Row_index']
            if not matched.empty:
                row_idx_dispenser = matched.values[0]
                reward_delay[i_trial] = (row_idx_dispenser - row_idx_select) / sampling_rate
            else:
                reward_delay[i_trial] = None  # Set to None if no matching rows are found
            
            # Initialize onset
            row_idx_init = df_next['Row_index'].iloc[0]
            
            
            # Time calculations
            # Selection time
            selection_time[i_trial] = (row_idx_select - row_idx_stim_photodiode) / sampling_rate
                
            # Time investment
            time_investment[i_trial] = (row_idx_init - row_idx_select) / sampling_rate


            # Reaction time
            # RT Left
            tf = (df['Column_name_changed'].isin(['Left_button'])) & (df['Left_button'].isin([1])) & (df['State'].isin([3, 4]))
            if not df.loc[tf, 'Row_index'].empty:
                row_idx_RT_L = df.loc[tf, 'Row_index'].values[0]
                RT_left[i_trial] = (row_idx_RT_L - row_idx_stim_photodiode) / sampling_rate
            else:
                RT_left[i_trial] = None  # Set to None if no matching rows are found
            # RT Light
            tf = (df['Column_name_changed'].isin(['Right_button'])) & (df['Right_button'].isin([1])) & (df['State'].isin([3, 4]))
            if not df.loc[tf, 'Row_index'].empty:
                row_idx_RT_R = df.loc[tf, 'Row_index'].values[0]
                RT_right[i_trial] = (row_idx_RT_R - row_idx_stim_photodiode) / sampling_rate
            else:
                RT_right[i_trial] = None  # Set to None if no matching rows are found
                
            
            # Push count left and right (Choice)
            df_push_count = df.loc[(df["State"] == 3) | (df["State"] == 4)]

            cnd1 = df_push_count["Column_name_changed"] == "Left_button"
            cnd2 = df_push_count["Left_button"] == 1
            PC_left[i_trial] = len(df_push_count.loc[cnd1 & cnd2])

            cnd1 = df_push_count["Column_name_changed"] == "Right_button"
            cnd2 = df_push_count["Right_button"] == 1
            PC_right[i_trial] = len(df_push_count.loc[cnd1 & cnd2])
            
            
            # Push count left and right (Initialization)
            df_push_count = df.loc[(df["State"] >= 5) ]
            cnd1 = df_push_count["Column_name_changed"] == "Left_button"
            cnd2 = df_push_count["Left_button"] == 1
            IPC_left[i_trial] = len(df_push_count.loc[cnd1 & cnd2])

            cnd1 = df_push_count["Column_name_changed"] == "Right_button"
            cnd2 = df_push_count["Right_button"] == 1
            IPC_right[i_trial] = len(df_push_count.loc[cnd1 & cnd2])


    
    # Session meta data
    subject, program_name, descriptions, start_time, end_time = extract_metadata(meta_file)
    
    # Status
    status = np.full_like(correct, 'Other', dtype=object)
    status[np.logical_and(correct, omission)] = 'Probe'
    status[np.logical_and(~correct, omission)] = 'Error'
    status[np.logical_and(correct, reward)] = 'Correct'

    # Create DataFrame
    trials = pd.DataFrame({ 'Subject': subject,
                            'Program_name': program_name,
                            'Start_datetime': start_time,
                            'End_datetime': end_time,
                            'Descriptions': descriptions,
                            'Block': block,
                            'Trial': n_trial, 
                            'Stim': stim, 
                            'Cue_left': cue_left, 
                            'Cue_right': cue_right, 
                            'Contrast': contrast, 
                            'Choice': choice, 
                            'Correct': correct,
                            'Delay': delay,
                            'Reward': reward,
                            'Omission': omission,
                            'Status': status,
                            'Stim_delay': stim_delay,
                            'Reward_delay': reward_delay,
                            'RT_left': RT_left,
                            'RT_right': RT_right,
                            'Selection_time': selection_time,
                            'Time_investment': time_investment,
                            'PC_left': PC_left,
                            'PC_right': PC_right,
                            'IPC_left': IPC_left,
                            'IPC_right': IPC_right,
                            })
    
    # Add conditional probabilities
    cnd_prob_prestim = ana_behv_sglssn.conditional_probabilities_prestim(trials, alter_index_border=0.8)
    trials["Debias"] = cnd_prob_prestim["alter_counts_index"]
    
    # Add recording 
    tf_description = "rec" in descriptions.strip()
    trials["Recording"] = int(tf_description)
    
    # Contrast list
    df = trials.copy(deep=True)
    df = df.dropna(subset=['Trial'])
    trials["Num_diff_contrast"] = np.abs(df['Contrast']).unique().shape[0] 
    
    # Chronometric function
    rho, p_value, h = ana_behv_sglssn.chronometric_function_corrlation(trials, stat="spearman", sig_boarder=0.05)
    trials["Chronofunc_pvalue"] = p_value
    
    return trials




if __name__ == "__main__":
    root_folder = 'Z:/Data'
    target_subjects = ['RSS023', 'RSS025', 'RSS026', 'RSS027', 'RSS030', 'RSS033', 'RSS036', 'RSS038', 'RSS039', 'RSS040', 'RSS041', 'RSS044', 'RSS045']
    #target_subjects = ['RSS044', 'RSS045']
    generate_trials_auto(root_folder, target_subjects)