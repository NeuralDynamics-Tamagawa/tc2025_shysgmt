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
from typing import List, Optional


# Original 
import libs.utils.utils as utils
import libs.analysis.behavior_single_session as ana_behv_sglssn



def diff_time_hour(trials):
    time_diff = trials['End_datetime'].values[0] - trials['Start_datetime'].values[0]
    time_diff_seconds = time_diff / np.timedelta64(1, 's')  
    time_diff_float = time_diff_seconds / 3600  
    return round(time_diff_float, 2)  



def generate_session_information(trials_session):
    keys = list(trials_session.groups.keys())

    session_info = pd.DataFrame() 

    for i_key in range(len(keys)):
        
        # Load trials dataset
        key = keys[i_key]
        trials = trials_session.get_group(key).copy()
        
        # Calculate session information
        # Calculate conditional probability
        cnd_prob_prestim = ana_behv_sglssn.conditional_probabilities_prestim(trials, alter_index_border=0.8)
        # Calculate correct rate
        correct_rate = trials.loc[trials['Contrast'].isin([-1, 1]), 'Correct'].mean()
        # Calculate chronometric function correlation
        trials_target = trials[(trials['Time_investment']>1.5)&(trials['Time_investment']<20)]
        rho, p_value, h = ana_behv_sglssn.chronometric_function_corrlation(trials_target, stat="spearman", sig_boarder=0.05)
        
        if trials['Program_name'].values[0] == 'dm2afc_illusion_of_control_v001':
            corr_TI_ST = ana_behv_sglssn.correlation_TI_ST(trials, outliner_cut_method=("percentile", 99), sig_border=0.05)
        else:
            corr_TI_ST = {
                "Probe": {"R":0, "p":1, "h":0},
                "Error": {"R":0, "p":1, "h":0},
            } 
    
        # 
        description = trials['Descriptions'].values[0]
        tf_description = "rec" in description.strip()
        
        
        session_info.loc[i_key, 'Subject']             = trials['Subject'].values[0]
        session_info.loc[i_key, 'Session_no']          = int(trials['Session_no'].values[0])
        session_info.loc[i_key, 'Program_name']        = trials['Program_name'].values[0]
        session_info.loc[i_key, 'Num_diff_contrast']   = int(trials['Num_diff_contrast'].values[0])
        session_info.loc[i_key, 'Training_time_hour']  = diff_time_hour(trials)
        session_info.loc[i_key, 'Trials_per_min']      = trials.shape[0]/diff_time_hour(trials)/60
        session_info.loc[i_key, 'Num_trials']          = trials.shape[0]
        session_info.loc[i_key, 'Debias']              = cnd_prob_prestim["alter_counts_index"]
        session_info.loc[i_key, 'Alter_rate']          = cnd_prob_prestim["alter_rate"]
        session_info.loc[i_key, 'Correct_rate']        = correct_rate
        session_info.loc[i_key, 'Corr_TI_ST_Probe']    = corr_TI_ST["Probe"]["R"]
        session_info.loc[i_key, 'Corr_TI_ST_Error']    = corr_TI_ST["Error"]["R"]
        session_info.loc[i_key, 'Chronofunc_corr']     = rho
        session_info.loc[i_key, 'Chronofunc_corr_p']   = p_value
        session_info.loc[i_key, 'Chronofunc_corr_sig'] = int(h)
        session_info.loc[i_key, 'Recording']           = int(tf_description)
        session_info.loc[i_key, 'Start_datetime']      = trials['Start_datetime'].values[0]
        
    program_mapping = {'dm2afc_gc_v001': 1, 
                       'dm2afc_cousor_v001': 2,
                       'dm2afc_cursor_v001': 2,
                       'dm2afc_illusion_of_control_v001': 3,
                       'dm2afc_illusion_of_control_v001': 4}
    
    session_info['Program_name'] = session_info['Program_name'].map(program_mapping)
    
    return session_info



def psychometric_function(trials_pooled, groupby_keys=['Subject']):
    
    trials_pooled_grouped = trials_pooled.groupby(groupby_keys)
    keys =list(trials_pooled_grouped.groups.keys()) 
    
    psyfun = pd.DataFrame()
    
    for key in keys:
        
        trials = trials_pooled_grouped.get_group(key)
        df = trials[['Contrast', 'Choice']].copy()
        df['Choice'] = (df['Choice'] == 'right').astype(int)

        df = df.groupby('Contrast', as_index=False).mean()
        
        for i_gkey in range(0, len(groupby_keys)):
            df[groupby_keys[i_gkey]] = key[i_gkey]

        psyfun = pd.concat([psyfun, df], ignore_index=True)

    return psyfun




def compute_psyfun(
    trials_pooled: pd.DataFrame,
    groupby_keys: List[str] = ['Subject', 'Start_datetime', 'Contrast'],
    ) -> pd.DataFrame:
    
    # Copy the trials_pooled and maping the Choice column to pRight
    df = trials_pooled.copy()
    df['Right'] = df['Choice'].map({'left': 0, 'right': 1}).astype(int)
    
    # Compute the conditional probability
    psyfun = df.groupby(groupby_keys, observed=True).agg(
            pRight=('Right', 'mean'),
            Count =('Right', 'size'),
            ).reset_index()
    
    return psyfun




def compute_conditional_psyfun(
    trials_pooled: pd.DataFrame,
    groupby_keys: List[str] = ['Subject', 'Start_datetime'],
    column: str = 'Time_investment',
    bins:List[float] = [0, 5, 1],
    ) -> pd.DataFrame:
    
    # Copy the trials_pooled and maping the Choice column to pRight
    df = trials_pooled.copy()
    df['Right'] = df['Choice'].map({'left': 0, 'right': 1}).astype(int)
    
    # Generate groupby keys
    groupby_keys = groupby_keys+['Contrast', 'Condition']
    
    # Create a new column 'Condition' based on the specified column
    if bins == None:
        df['Condition'] = df[column]
    else:
        bins[1] = bins[1] + 1e-10
        bins = np.arange(*bins)
        df['Condition'] = pd.cut(df[column], bins=bins).apply(lambda x: x.right)
        
    # Compute the conditional probability
    psyfun = df.groupby(groupby_keys, observed=True).agg(
            pRight=('Right', 'mean'),
            Count =('Right', 'size'),
            ).reset_index()
    
    return psyfun




def compute_calibration_curves(
    trials_pooled: pd.DataFrame,
    groupby_keys: List[str] = ['Subject', 'Start_datetime'],
    column: str = 'Contrast',
    bins_ti:List[float] = [0, 10, 1],
    ) -> pd.DataFrame:
    
    # Copy the trials_pooled and maping the Choice column to pRight
    df = trials_pooled.copy()
    df['Accuracy'] = df['Correct'].astype(int)
    bins = np.arange(*bins_ti)
    #df['Binned_time_investment'] = pd.cut(df['Time_investment'], bins=bins).apply(lambda x: x.right)
    df['Binned_time_investment'] = pd.cut(df['Time_investment'], bins=bins).apply(lambda x: (x.left + x.right) / 2)
    
    # Generate groupby keys
    groupby_keys = groupby_keys+['Binned_time_investment']
    
    # Create a new column 'Condition' based on the specified column
    if column is not None:
        df['Condition'] = df[column]
        groupby_keys = groupby_keys+['Condition']
        
    # Compute the conditional probability
    calibration_curves = df.groupby(groupby_keys, observed=True).agg({
                            'Accuracy': "mean",
                            }).reset_index()
        
    return calibration_curves




def compute_conditional_RTST(
    trials_pooled: pd.DataFrame,
    groupby_keys: List[str] = ['Subject', 'Start_datetime'],
    conditional_column: str = 'Correct',
    bins:List[float] = [0, 10, 1],
    ) -> pd.DataFrame:
    
    # Copy the trials_pooled and maping the Choice column to pRight
    df = trials_pooled.copy()
    
    # Generate groupby keys
    groupby_keys = groupby_keys+['Contrast', 'Condition']
    
    # Create a new column 'Condition' based on the specified column
    if bins == None:
        df['Condition'] = df[conditional_column]
    else:
        bins[1] = bins[1] + 1e-10
        bins = np.arange(*bins)
        df['Condition'] = pd.cut(df[conditional_column], bins=bins).apply(lambda x: x.right)
        
    # Compute the conditional probability
    RTST = df.groupby(groupby_keys, observed=True).agg(
            Selection_time=('Selection_time', 'mean'),
            Reaction_time=('Reaction_time', 'mean'),
            Count =('Selection_time', 'size'),
            ).reset_index()

    return RTST
    



def compute_conditional_vevaiometric_curves(
    trials_pooled: pd.DataFrame,
    groupby_keys: List[str] = ['Subject', 'Start_datetime'],
    column: str = 'Selection_time',
    bins:List[float] = [0, 10, 1],
    ) -> pd.DataFrame:
    
    # Copy the trials_pooled and maping the Choice column to pRight
    df = trials_pooled.copy()
    
    # Generate groupby keys
    groupby_keys = groupby_keys+['Contrast', 'Status',  'Condition']
    
    # Create a new column 'Condition' based on the specified column
    if bins == None:
        df['Condition'] = df[column]
    else:
        bins[1] = bins[1] + 1e-10
        bins = np.arange(*bins)
        df['Condition'] = pd.cut(df[column], bins=bins).apply(lambda x: x.right)
        
    # Compute the conditional probability
    TI = df.groupby(groupby_keys, observed=True).agg(
            Time_investment=('Time_investment', 'mean'),
            Count =('Time_investment', 'size'),
            ).reset_index()
    
    return TI
    




def contrast_TI(trials_pooled, groupby_keys=['Subject'], calc_method="mean"):
    
    trials_pooled_grouped = trials_pooled.groupby(groupby_keys)
    keys =list(trials_pooled_grouped.groups.keys()) 
    
    contrast_TI = pd.DataFrame()
    
    for key in keys:
        
        trials = trials_pooled_grouped.get_group(key)
        df = trials[["Contrast", "Status", "Time_investment"]].copy()
        df = df[~df["Status"].isin(["Correct", "Other"])]

        if calc_method == "mean":
            df = df.groupby(['Contrast', 'Status'], as_index=False).mean()
        elif calc_method == "median":
            df = df.groupby(['Contrast', 'Status'], as_index=False).median()
        
        for i_gkey in range(0, len(groupby_keys)):
            df[groupby_keys[i_gkey]] = key[i_gkey]

        contrast_TI = pd.concat([contrast_TI, df], ignore_index=True)
        

    return contrast_TI



def chronometric_function(trials_pooled, groupby_keys=['Subject'], calc_method="mean"):
    
    trials_pooled_grouped = trials_pooled.groupby(groupby_keys)
    keys =list(trials_pooled_grouped.groups.keys()) 
    
    chronometric_func = pd.DataFrame()
    
    for key in keys:
        
        trials = trials_pooled_grouped.get_group(key)
        
        df = trials[["Contrast", "Status", "Time_investment"]].copy()
        df = df[df["Status"] != "Correct"]
        abscont = df['Contrast'].abs() 
        ce = df["Status"].apply(lambda x: 1 if x == "Probe" else -1)
        evidence = abscont * ce
        df["Evidence"]  = evidence
        df = df.dropna()
        df.drop(columns=["Contrast", "Status"], inplace=True)
        
        
        if calc_method == "mean":
            df = df.groupby('Evidence', as_index=False).mean()
        elif calc_method == "median":
            df = df.groupby('Evidence', as_index=False).median()
        
        
        for i_gkey in range(0, len(groupby_keys)):
            df[groupby_keys[i_gkey]] = key[i_gkey]

        chronometric_func = pd.concat([chronometric_func, df], ignore_index=True)

    return chronometric_func



def contrast_TI_old(trials_pooled_grouped):
    keys = trials_pooled_grouped.groups.keys()
    
    contrast_TI = pd.DataFrame()
    for i_k in range(len(keys)):
        key = list(keys)[i_k]
        
        trials = trials_pooled_grouped.get_group(key)
        df = trials[["Contrast", "Status", "Time_investment"]].copy()
        df = df[~df["Status"].isin(["Correct", "Other"])]

        df = df.groupby(['Contrast', 'Status'], as_index=False).mean()
        
        
        df['Subject'] = key[0]
        df['Session'] = key[1]

        contrast_TI = pd.concat([contrast_TI, df], ignore_index=True)
    
    return contrast_TI