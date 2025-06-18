import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Optional





def flag_pseudo_probe(status, pseudo_prob=1/8):
    if status == 'Probe':
        return 1
    elif status == 'Correct':
        return 0
    elif status == 'Error':
        return 1 if np.random.rand()<float(pseudo_prob) else 0
    elif status == 'Other':
        return 0
    else:
        return np.nan
    
    



def psychometric_curve(dataframe):
    
    is_dict = isinstance(dataframe, dict)
    
    # Grouped data
    if is_dict:
        print
        keys = list(dataframe.keys())

        psyfunc = []  
        for i_key in range(len(keys)):
            trials = dataframe[keys[i_key]].copy()
            trials['pRightward'] = (trials['Choice'] == 'right').astype(int)
            df = trials.groupby("Contrast", as_index=False).agg({
                "pRightward": "mean",
                "Start_datetime": "first",
                "Subject": "first"
            })
            psyfunc.append(df)

        psyfunc = pd.concat(psyfunc, ignore_index=True)
    
    # Single session data
    else:
        trials = dataframe.copy()
        trials['pRightward'] = (trials['Choice'] == 'right').astype(int)
        df = trials.groupby("Contrast", as_index=False).agg({
            "pRightward": "mean",
            "Start_datetime": "first",
            "Subject": "first"
        })
    
    return psyfunc


def conditional_probabilities_prestim(trials, alter_index_border=0.8):
    
    df = trials[["Stim", "Correct"]].copy()
    
    # Add pre stim and correct 
    df["Correct"] = df["Correct"].map({True: "Correct", False: "Error"})
    df["Prev_Stim"] = df["Stim"].shift(1)
    df["Prev_Correct"] = df["Correct"].shift(1)
    
    # Drop NaN
    df = df.dropna().reset_index(drop=True)
    
    # Calc coditional probabilities & Complement (the missing data)
    expected_index = pd.MultiIndex.from_product([["left", "right"], ["Correct", "Error"]], names=["Prev_Stim", "Prev_Correct"])
    expected_columns = ["left", "right"]  
    counts = df.groupby(["Prev_Stim", "Prev_Correct", "Stim"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=expected_index, columns=expected_columns, fill_value=0)
    
    correct = df[df["Correct"] == "Correct"].groupby(["Prev_Stim", "Prev_Correct", "Stim"]).size().unstack(fill_value=0)
    err = df[df["Correct"] == "Error"].groupby(["Prev_Stim", "Prev_Correct", "Stim"]).size().unstack(fill_value=0)
    rate = correct.div(counts).fillna(0)
    
    # Calc alter_counts and alter_counts_index
    counts = np.array(counts)
    alter = counts[[1, 2], 0].sum() + counts[[0, 3], 1].sum()
    continuous = counts[[0, 3], 0].sum()+counts[[1, 2], 1].sum()
    alter_counts = (alter)/(alter+continuous)
    if alter_counts > alter_index_border:
        alter_counts_index = 1
    else:
        alter_counts_index = 0
        
    #Calc alter_rate and alter_rate_index
    rate = np.array(rate)
    alter = rate[[1, 2], 0].sum() + rate[[0, 3], 1].sum()
    continuous = rate[[0, 3], 0].sum()+rate[[1, 2], 1].sum()
    alter_rate = (alter-continuous)/(alter+continuous)
    
    rate = pd.DataFrame(rate, index=expected_index, columns=expected_columns)
    
    
    # 
    result = {
        "counts": counts,
        "correct": correct,
        "err": err,
        "rate": rate,
        "alter_counts": alter_counts,
        "alter_counts_index": alter_counts_index,
        "alter_rate": alter_rate
    }
    
    return result




def compute_calibration_curves(
    trials: pd.DataFrame,
    bins_ti:List[float] = [0, 10, 1],
    slide: float = 0.1,
    n_shuffles: int = 1000
    ) -> pd.DataFrame:
    
    trials = trials.copy()
    trials['Contrast'] = trials['Contrast'].abs()
    trials['Accuracy'] = trials['Correct'].astype(int)
    
    # Copy the trials_pooled and maping the Choice column to pRight
    calibration_curves_shuf = pd.DataFrame()
    
    n_slide = int((bins_ti[1] - bins_ti[0]) / slide)
    for i_range in range(0, n_slide):
        
        slide = 0.1 * i_range
        bins_ti_slided = [bins_ti[0] + slide, bins_ti[1] + slide, bins_ti[2]]
        bins = np.arange(*bins_ti_slided)
        
        for i in range(1, n_shuffles): 
            df = trials.copy()
            df['Binned_time_investment'] = pd.cut(df['Time_investment'], bins=bins).apply(lambda x: (x.left + x.right) / 2)
            df['Pseudo_probe'] = df['Status'].apply(lambda status: flag_pseudo_probe(status, pseudo_prob=1/8))
            df = df[df['Pseudo_probe']==1]
            cc = df.groupby(['Contrast', 'Binned_time_investment'], observed=True).agg({
                    'Accuracy': "mean",
                    }).reset_index()
            calibration_curves_shuf = pd.concat([calibration_curves_shuf, cc], ignore_index=True)
        
        df = calibration_curves_shuf.copy()
        calibration_curves = df.groupby(['Contrast', 'Binned_time_investment'], observed=True).agg({
                                "Accuracy": "mean",
                                }).reset_index()
        
    return calibration_curves, calibration_curves_shuf



def chronometric_function_corrlation(trials, stat="pearson", sig_boarder=0.05):
    # Prepare data
    df = trials[["Contrast", "Status", "Time_investment"]].copy()
    df = df[df["Status"] != "Correct"]
    abscont = df['Contrast'].abs() 
    ce = df["Status"].apply(lambda x: 1 if x == "Probe" else -1)
    evidence = abscont * ce
    df["Evidence"]  = evidence
    df = df.dropna()
    
    # Calc correlation
    if evidence.unique().size >= 2:
        if stat == "pearson":
            rho, p_value = stats.pearsonr(df["Evidence"], df["Time_investment"])
        elif stat == "spearman":
            rho, p_value = stats.spearmanr(df["Evidence"], df["Time_investment"])
    else:
        rho = 0
        p_value = 1
    
    # Hypothesis testing
    if p_value < sig_boarder:
        h = 1
    else:
        h = 0
    
    return rho, p_value, h



# co
def correlation_TI_ST(trials, outliner_cut_method = ("percentile", 95), sig_border = 0.05):
        # Copy the data
    df = trials.copy()
    df = df[df["Status"] != "Correct"]

    # Delete outliers
    TI = df["Time_investment"].dropna().to_numpy()
    ST = df["Selection_time"].dropna().to_numpy()
    
    # Outliner cut
    border = outliner_cut_method[1]
    if outliner_cut_method[0] == "percentile":
        TI_threshold = np.percentile(TI, border)
        ST_threshold = np.percentile(ST, border)
    elif outliner_cut_method[0] == "std":
        TI_threshold = TI.mean() + TI.std() * border
        ST_threshold= ST.mean() + ST.std() * border
    else:
        print("Error: Invalid outliner cut method")
    df = df[df["Time_investment"] < TI_threshold]
    df = df[df["Selection_time"] < ST_threshold]
    


    # Correlation calculation for "Probe" data
    df_probe = df[df["Status"] == "Probe"]
    rho_probe, p_probe = stats.spearmanr(df_probe["Selection_time"], df_probe["Time_investment"])

    # Correlation calculation for "Error" data
    df_error = df[df["Status"] == "Error"]
    rho_error, p_error = stats.spearmanr(df_error["Selection_time"], df_error["Time_investment"])
    
    if p_probe < sig_border:
        h_probe = 1
    else:
        h_probe = 0
        
    if p_error < sig_border:
        h_error = 1
    else:
        h_error = 0
        
    
    result = {
        "Probe": {"R":rho_probe, "p":p_probe, "h":h_probe},
        "Error": {"R":rho_error, "p":p_error, "h":h_error},
    }
    
    return result