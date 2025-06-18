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
import libs.utils.utils as utils
import libs.prep_behavior.generate_trials as gen_trials
import libs.visualization.behavior_single_session as plt_behv_snglssn 
import libs.submodules.psychofit as psy
import libs.handle.dataset as handle_dataset
import libs.analysis.behavior_single_session as ana_behv_sglssn


"""
==================== Plotting functions 1 ====================
- psychometoric functions
- reaction time
- time investment distoribuion
- vevaiometric curve (x=contrast y=time investment)
- calibration curve
- vevaiometric curve
- pshycometric functions with time investment
- reaction time vs trial
==============================================================
"""

color_LR = {       
    "left": (1.0, 0.5, 0.05), 
    "right": (0.6, 0.4, 0.7)  
}

color_PEC = {
    "Probe": (0.2, 0.3, 0.5),
    "Error": (0.7, 0.3, 0.2),
    "Correct": (0.2, 0.7, 0.2) 
}

color_HE = {
    "Hard": (0.8, 0.2, 0.8),    
    "Easy": (0.5, 0.2, 0.5)
}

color_SL = {
    "Short": (0.6, 0.8, 0.9),
    "Long": (0.2, 0.2, 0.8),
}








    


def correct_rate(ax, session_information, linewidth=2, fontsize=12, color_pallet="deep"):
    
    df = session_information.copy()
    df["Session_order"] = df.groupby("Subject").cumcount()
    unique_subjects = df['Subject'].unique()
    subject_mapping = {subj: val for subj, val in zip(unique_subjects, np.linspace(0, 0.3, len(unique_subjects)))}
    df["Subject_mapped"] = df["Subject"].map(subject_mapping)
    rec = df[df['Recording'].isin([1])]
    brc = df[(df['Program_name'].isin([3]))&(df['Num_diff_contrast'].isin([5]))]
    pretrn = df[(df['Program_name'].isin([1, 2]))&(df['Debias'].isin([1]))]
    
    sns.lineplot(
        data=df,
        x='Session_order',
        y='Correct_rate',
        hue='Subject',
        alpha=0.2,
        palette=color_pallet,
        legend=False,
        linewidth=linewidth,
        ax=ax,
    )
    sns.scatterplot(
        data=rec,
        x='Session_order',
        y='Subject_mapped',
        hue='Subject',
        alpha=1,
        palette=color_pallet,
        legend=False,
        ax=ax,
    )
    sns.lineplot(
        data=brc,
        x='Session_order',
        y='Subject_mapped',
        hue='Subject',
        alpha=0.5,
        palette=color_pallet,
        legend=False,
        linewidth=linewidth,
        ax=ax,
    )
    sns.lineplot(
        data=pretrn,
        x='Session_order',
        y='Correct_rate',
        hue='Subject',
        alpha=0.9,
        palette=color_pallet,
        legend=False,
        linewidth=linewidth,
        ax=ax,
    )
    sns.lineplot(
        data=df,
        x='Session_order',
        y='Correct_rate',
        estimator=np.mean,
        color='black',
        alpha=1,
        linewidth=linewidth,
        errorbar=None,
        ax=ax,
    )

    utils.biofig(linewidth=linewidth, fontsize=fontsize)
    plt.xticks(np.arange(0, 80, 10))
    plt.yticks(np.arange(0, 1.1, 0.5))
    plt.xlabel('Session')
    plt.ylabel('Correct rate \n (Δcontrast = ±1.0)')
    
    
     
