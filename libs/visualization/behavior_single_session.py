import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import libs.utils.utils as utils
from datetime import datetime
import itertools
import pickle
import libs.handle.dataset as handle_dataset

"""
==================== Plotting functions ====================
- Main 
    - Auto plotting behavior summary 

- Summary
    Auto plotting behavior summary (Cursor and Illusion of control)
    
- Metadata
- Plotting summary of the behavior (cursor)
- Plotting summary of the behavior (illusion of control)

- Single figure
    - Psychometoric functions
    - Reaction time
    - Time investment distoribuion
    - Time investment violinplot
    - Vevaiometric curve (x=contrast y=time investment)
    - Calibration curve
    - Vevaiometric curve
    - Pshycometric functions with time investment
    - Reaction time trial
    - correct rate
    - conditional probability
    - Correlation between time investment and selection time
    - Chronometric

-
============================================================
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



    

def plot_auto(root_folder, output_dir=None, target_subject=None):
    
    subject_folders = utils.get_subfolders(root_folder)
    
    # Extract target subjects
    if target_subject != None:
        subject_folders = subject_folders[subject_folders['folder_name'].isin(target_subject)]
 
    # Loop for each subject
    for i_subject in range(0, len(subject_folders)):
        session_folders = utils.get_subfolders(subject_folders.iloc[i_subject] ['absolute_path'])

        # Loop for each session
        for i_session in range(0, len(session_folders)):
            session_dir = session_folders.loc[i_session]['absolute_path']

            session = handle_dataset.Session(session_dir)
            utils.printlg(f"File: {session.session_dir}")
            
            if output_dir == None: 
                procTeensy_dir = os.path.join(session.session_dir, 'procTeensy')
                plot(session, output_dir = procTeensy_dir, savefig = True, pltshow=False)
            else:
                procTeensy_dir = os.path.join(session.session_dir, 'procTeensy')
                plot(session, output_dir = procTeensy_dir, savefig = True, pltshow=False)
                plot(session, output_dir = output_dir, savefig = True, pltshow=False)
            
            
            



def plot(session, output_dir = None, savefig = True, pltshow=False):
    meta_file = session.metaFile
    trials = session.trials
    trials = trials.dropna(subset=['Trial'])
    if output_dir == None: 
        output_dir = session.session_dir
    summary(meta_file, trials, output_dir, savefig=savefig, pltshow=pltshow)
    


def summary(meta_file, trials, output_dir, savefig=True, pltshow=False):
    
    header = meta_file['Header']
    program_name = header.loc[header["Name"] == "programName", "Value"] 

    if program_name.values[0] == 'dm2afc_gc_v001':
        cursor(meta_file, trials, output_dir, savefig=savefig, pltshow=pltshow)
    elif program_name.values[0] == 'dm2afc_cousor_v001':
        cursor(meta_file, trials, output_dir, savefig=savefig, pltshow=pltshow)
    elif program_name.values[0] == 'dm2afc_cursor_v001':
        cursor(meta_file, trials, output_dir, savefig=savefig, pltshow=pltshow)
    elif program_name.values[0] == 'dm2afc_illusion_of_control_v001':
        illusion_of_control(meta_file, trials, output_dir, savefig=savefig, pltshow=pltshow)
    elif program_name.values[0] == 'dm2afc_illusion_of_control_v002':
        illusion_of_control(meta_file, trials, output_dir, savefig=savefig, pltshow=pltshow)
    else:
        print("Error")
        
        
        
def metadata(meta_file):
    header = meta_file["Header"]
    time = meta_file["Time"]
    
    data_folder = header.loc[header["Name"] == "dataFolder", "Value"].values[0]
    descriptions = header.loc[header["Name"] == "descriptions", "Value"].values[0]
    
    datetime = time.loc[time["Name"] == "Datetime", "Value"].values[0]
    start_time = time.loc[time["Name"] == "Start", "Value"].values[0]
    end_time = time.loc[time["Name"] == "End", "Value"].values[0]
    
    session_duration = utils.parse_time(end_time, datetime) - utils.parse_time(start_time, datetime)

    return data_folder, descriptions, start_time, session_duration



# Plotting summary of the behavior (cursor)
def cursor(meta_file, trials, output_dir, savefig=True, pltshow=False):
    
    import matplotlib
    matplotlib.rcParams["font.family"] = "Arial"

    lw = 1
    fs = 8
    row = 4
    col = 6

    df = trials.copy()
    

    fig = plt.figure(figsize=(col*2.4, row*2))
    gs = fig.add_gridspec(row, col)  # 4行6列のグリッドを作成

    ax = fig.add_subplot(gs[0, 2:5])
    correct_rate(ax=ax, trials=df, linewidth=lw, fontsize=fs)
    
    ax = fig.add_subplot(gs[0, 5])
    gabor_contrast_map(ax=ax, trials=df, linewidth=lw, fontsize=fs)

    ax = fig.add_subplot(gs[:, 1])
    reaction_time_trials(ax=ax, trials=df, linewidth=lw, fontsize=fs, mrksize=20)

    ax = fig.add_subplot(gs[1, 2])
    psychometric_curve(ax=ax, trials=df, linewidth=lw, fontsize=fs)

    ax = fig.add_subplot(gs[2, 2])
    reaction_time(ax=ax, trials=df, linewidth=lw, fontsize=fs)
    #plt_behv.selection_time(ax=ax, df_trials=df, linewidth=lw, fontsize=fs)

    ax = fig.add_subplot(gs[2, 3])
    conditional_prob_prestim(ax1=ax, trials=df, linewidth=lw, fontsize=fs, mrksize=5)


    data_folder, descriptions, start_time, session_duration = metadata(meta_file)
    fig.suptitle(f"{data_folder}\n"
                 f"{descriptions}\n"
                 f"{start_time}\n"
                 f"{session_duration}",
                fontsize=fs, fontweight="bold", y=1.05, horizontalalignment='center')

    header = meta_file['Header']
    folder_name = header.loc[header['Name'] == "dataFolder", "Value"].values[0]
    file_name = os.path.join(output_dir, folder_name+".png")
    
    if savefig:
        plt.savefig(file_name, dpi=600, bbox_inches="tight")
        
    if pltshow:
        plt.show()
    else:
        plt.close()



# Plotting summary of the behavior (illusion of control)
def illusion_of_control(meta_file, trials, output_dir, savefig=True, pltshow=False):
    
    
    df = trials.copy()
    
    # Cut outliner 
    TI = df["Time_investment"].dropna().to_numpy().astype(float)
    TI_threshold = np.percentile(TI, 99)
    df1 = df[df["Time_investment"] < TI_threshold]
    
    # Target trials Time investment(TI) 1.5 < TI < 15
    df2 = df[(df["Time_investment"] > 1.5) & (df["Time_investment"] < 15)]
    
    
    
    
    import matplotlib
    matplotlib.rcParams["font.family"] = "Arial"

    lw = 1
    fs = 8
    row = 4
    col = 6

    

    fig = plt.figure(figsize=(col*2.4, row*2))
    gs = fig.add_gridspec(row, col)  # 4行6列のグリッドを作成

    ax = fig.add_subplot(gs[0, 2:5])
    correct_rate(ax=ax, trials=df, linewidth=lw, fontsize=fs)
    
    ax = fig.add_subplot(gs[0, 5])
    gabor_contrast_map(ax=ax, trials=df, linewidth=lw, fontsize=fs)


    ax = fig.add_subplot(gs[:, 1])
    reaction_time_trials(ax=ax, trials=df, linewidth=lw, fontsize=fs, mrksize=20)

    ax = fig.add_subplot(gs[1, 2])
    psychometric_curve(ax=ax, trials=df, linewidth=lw, fontsize=fs)

    ax = fig.add_subplot(gs[1, 3])
    calibration_curve(ax=ax, trials=df2, linewidth=lw, fontsize=fs, HE_border=0.25)

    ax = fig.add_subplot(gs[1, 4])
    vevaiometric_curve(ax=ax, trials=df2, linewidth=lw, fontsize=fs)

    ax = fig.add_subplot(gs[1, 5])
    psychometric_curve_condition_TI(ax=ax, trials=df, linewidth=lw, fontsize=fs, border_time = 5)

    ax = fig.add_subplot(gs[2, 2])
    reaction_time(ax=ax, trials=df, linewidth=lw, fontsize=fs)
    #selection_time(ax=ax, trials=df, linewidth=lw, fontsize=fs)

    ax = fig.add_subplot(gs[2, 3])
    conditional_prob_prestim(ax1=ax, trials=df, linewidth=lw, fontsize=fs, mrksize=5)

    ax = fig.add_subplot(gs[2, 4])
    vevaiometric_curve_contrast(ax=ax, trials=df2, linewidth=lw, fontsize=fs)

    ax = fig.add_subplot(gs[2, 5])
    time_investment_distoribuion(ax=ax, trials=df, linewidth=lw, fontsize=fs)

    ax = fig.add_subplot(gs[3, 2])
    corr_TI_ST(ax=ax, trials=df, linewidth=lw, fontsize=fs, mrksize=5, outliner_cut_method=("percentile", 99))
    #corr_TI_ST(ax=ax, trials=df, linewidth=lw, fontsize=fs, mrksize=5)
    
    ax = fig.add_subplot(gs[3, 3])
    chronometric_function(ax=ax, trials=df2, linewidth=lw, fontsize=fs, mrksize=5)

    ax = fig.add_subplot(gs[3, 4])
    vevaiometric_curve_contrast_pushcount(ax=ax, trials=df, linewidth=lw, fontsize=fs, ylim =[0.5, 3])
    
    ax = fig.add_subplot(gs[3, 5])
    push_count_heatmap(ax, trials=df, linewidth=lw, fontsize=fs)
        

    data_folder, descriptions, start_time, session_duration = metadata(meta_file)
    fig.suptitle(f"{data_folder}\n"
                 f"{descriptions}\n"
                 f"{start_time}\n"
                 f"{session_duration}",
                fontsize=fs, fontweight="bold", y=1.05)

    header = meta_file['Header']
    folder_name = header.loc[header['Name'] == "dataFolder", "Value"].values[0]
    file_name = os.path.join(output_dir, folder_name+".png")
    
    if savefig:
        plt.savefig(file_name, dpi=600, bbox_inches="tight")
        
    if pltshow:
        plt.show()
    else:
        plt.close()



# Ploting stimulus condition 
def gabor_contrast_map(trials, ax=None, linewidth=3, fontsize=15):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6)) 
    
    # Create a copy of the DataFrame
    df = trials[['Cue_left','Cue_right']]
    df = df.groupby(['Cue_left','Cue_right']).size().reset_index(name='counts')
    
    sns.scatterplot(
    data=df,
    x='Cue_left',
    y='Cue_right',
    hue='counts',
    s=fontsize*5,
    legend=True,
    ax=ax
    )
    ax.legend(fontsize=fontsize/2)
    utils.biofig(linewidth=linewidth, fontsize=fontsize)
    # Set axis labels and style
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel("Right contrast", fontsize=fontsize)
    plt.ylabel("Left contrast", fontsize=fontsize)
    plt.title("Gabor contrast map", fontsize=fontsize)
    plt.grid(False)
    # Display the plot
    plt.tight_layout()



# Plotting psychometoric functions
def psychometric_curve(ax, trials, linewidth=3, fontsize=15):
    # Create a copy of the DataFrame
    df = trials[['Contrast', 'Choice']].copy()
    df['Choice'] = df['Choice'] == 'right'

    # Plot parameters
    custom_palette = {(0.2, 0.2, 0.2)}

    # Plot using seaborn
    sns.lineplot(
        data=df,
        x="Contrast", 
        y="Choice", 
        estimator=np.mean, 
        linewidth=linewidth,
        errorbar=("se", 1),
        color = 'black',
        ax=ax
    )
    utils.biofig(ax=ax, linewidth=linewidth, fontsize=fontsize)
    # Set axis labels and style
    plt.xlim([-1.1, 1.1])
    plt.ylim([0, 1])
    plt.xlabel("Contrast", fontsize=fontsize)
    plt.ylabel("pRightward", fontsize=fontsize)
    plt.title("Psychometric function", fontsize=fontsize)
    plt.grid(False)
    # Display the plot
    plt.tight_layout()



# Plotting reaction time
def reaction_time(ax, trials, linewidth=3, fontsize=15, color_LR=color_LR):
    # Create a copy of the DataFrame
    df = trials.copy()
    df["Reaction_time"] = df.apply(lambda row: row["RT_left"] if row["Choice"] == "left" else row["RT_right"], axis=1)
    status_order = ['left', 'right']
    df['Choice'] = pd.Categorical(df['Choice'], categories=status_order, ordered=True)
    #df = df[df["Reaction_time"] < 5].reset_index(drop=True)


    # Plot using seaborn
    sns.lineplot(
        data=df, 
        x="Contrast", 
        y="Reaction_time", 
        hue="Choice", 
        estimator=np.mean,
        errorbar=('se', 1), 
        linewidth=linewidth, 
        palette=color_LR,
        ax=ax
    )
    utils.biofig(ax=ax, linewidth=linewidth, fontsize=fontsize)
    # Set axis labels and style
    plt.xlim([-1.1, 1.1])
    plt.ylim([0, 5])
    plt.xlabel("Contrast", fontsize=fontsize)
    plt.ylabel("Reaction time (s)", fontsize=fontsize)
    plt.legend(title="", fontsize=fontsize, frameon=False)
    plt.title("Reaction time", fontsize=fontsize)
    plt.grid(False)
    # Display the plot
    plt.tight_layout()
    


# Plotting reaction time
def selection_time(ax, trials, linewidth=3, fontsize=15, color_LR=color_LR):
    # Create a copy of the DataFrame
    df = trials[['Contrast', 'Choice', 'Correct', 'Selection_time', 'Status']].copy()
    status_order = ['left', 'right']
    df['Choice'] = pd.Categorical(df['Choice'], categories=status_order, ordered=True)

    # Plot using seaborn
    sns.lineplot(
        data=df, 
        x="Contrast", 
        y="Selection_time", 
        hue="Choice", 
        estimator=np.mean,
        errorbar=('se', 1), 
        linewidth=linewidth, 
        palette=color_LR,
        ax=ax
    )
    utils.biofig(ax=ax, linewidth=linewidth, fontsize=fontsize)
    # Set axis labels and style
    plt.xlim([-1.1, 1.1])
    plt.ylim([0, 5])
    plt.xlabel("Contrast", fontsize=fontsize)
    plt.ylabel("Selection time (s)", fontsize=fontsize)
    plt.legend(title="", fontsize=fontsize, frameon=False)
    plt.title("Selection time", fontsize=fontsize)
    plt.grid(False)
    # Display the plot
    plt.tight_layout()
    
    

# Plotting time investment histogram 
def time_investment_distoribuion(ax, trials, linewidth=3, fontsize=15, color_PEC=color_PEC):
    # Create a copy of the DataFrame
    df = trials[['Time_investment', 'Status', 'Correct']].copy()
    status_order = ['Correct', 'Probe', 'Error']
    df['Status'] = pd.Categorical(df['Status'], categories=status_order, ordered=True)
    df = df[df["Status"] != 'Correct'].reset_index(drop=True)

    bins = np.arange(0, 21, 1)
    # Plot using seaborn
    for status, subset in df.groupby("Status", observed=False):
        sns.histplot(
            data=subset,
            x="Time_investment",
            stat="density", 
            bins=bins,
            kde=False,
            label=status,
            element="step",
            linewidth=linewidth,
            color=color_PEC.get(status, "black"),
            ax=ax
        )
    
    utils.biofig(ax=ax, linewidth=linewidth, fontsize=fontsize)
    plt.xlim([0, 21])
    plt.xticks(np.arange(0, 21, 5))
    plt.xlabel("Time investment (s)", fontsize=fontsize)
    plt.ylabel("Density", fontsize=fontsize)
    plt.title("TI histogram", fontsize=fontsize)
    plt.grid(False)
    plt.tight_layout()
    
    

# Plotting time investment histogram
def time_investment_violinplot(ax, trials, linewidth=3, fontsize=15, color_PEC=color_PEC):
    # Create a copy of the DataFrame
    df = trials[['Time_investment', 'Status', 'Correct']].copy()
    status_order = ['Correct', 'Probe', 'Error']
    df['Status'] = pd.Categorical(df['Status'], categories=status_order, ordered=True)

    color_PEC = {key: tuple(min(c + 0.1, 1.0) for c in value) for key, value in color_PEC.items()}
    
    sns.violinplot(
        x="Status",
        y="Time_investment",
        data=df,
        hue="Status",
        legend=False,
        linewidth=linewidth,
        palette=color_PEC,
    )
    
    utils.biofig(ax=ax, linewidth=linewidth, fontsize=fontsize)
    plt.xlim([-1, 4])
    plt.ylim([0, 31])
    #plt.xticks(np.arange(0, 4, 1))
    plt.xlabel("Status", fontsize=fontsize)
    plt.ylabel("Time investment (s)", fontsize=fontsize)
    plt.title("Time investment", fontsize=fontsize)
    plt.grid(False)
    plt.tight_layout()



# Plotting time investment vs contrast 
def vevaiometric_curve_contrast(ax, trials, linewidth=3, fontsize=15, estimator='mean', ylim =[0, 16], correct = True, color_PEC=color_PEC):
    # Create a copy of the DataFrame
    df = trials[['Contrast', 'Time_investment', 'Status']].copy()
    if correct == True:
        status_order = ['Correct', 'Probe', 'Error']
    else:
        status_order = ['Probe', 'Error']
        
    df['Status'] = pd.Categorical(df['Status'], categories=status_order, ordered=True)
    
    if estimator == 'mean':
        sns.lineplot(
        x="Contrast", 
        y="Time_investment", 
        hue="Status",
        data=df, 
        estimator=np.mean,
        errorbar=('se', 1), 
        linewidth=linewidth, 
        palette=color_PEC,
        ax=ax
    )
        
    elif estimator == 'median':
        sns.lineplot(
        x="Contrast", 
        y="Time_investment", 
        hue="Status",
        data=df, 
        estimator=np.median,
        errorbar=('pi', 50), 
        linewidth=linewidth, 
        palette=color_PEC,
        ax=ax
    )
    
    
    # Plot using seaborn
    
    utils.biofig(linewidth=linewidth, fontsize=fontsize)
    plt.xlim([-1.1, 1.1])
    plt.ylim(ylim)
    plt.xlabel("Contrast", fontsize=fontsize)
    plt.ylabel("Time investment (s)", fontsize=fontsize)
    plt.title("TI-contrast", fontsize=fontsize)
    plt.legend(title="", fontsize=fontsize, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    


def vevaiometric_curve_contrast_pushcount(ax, trials, linewidth=3, fontsize=15, ylim =[0.5, 3], color_PEC=color_PEC):
    # Create a copy of the DataFrame
    df = trials.copy()
    df["Push_count_choice"] = df.apply(lambda row: row["PC_left"] if row["Choice"] == "left" else row["PC_right"], axis=1)
    status_order = ['Correct', 'Probe', 'Error']
    df['Status'] = pd.Categorical(df['Status'], categories=status_order, ordered=True)
    
    sns.lineplot(
        x="Contrast",
        y="Push_count_choice", 
        hue="Status",
        data=df, 
        estimator=np.mean,
        errorbar=('se', 1), 
        linewidth=linewidth, 
        palette=color_PEC,
        ax=ax
    )
        
    # Plot using seaborn
    utils.biofig(linewidth=linewidth, fontsize=fontsize)
    plt.xlim([-1.1, 1.1])
    plt.ylim(ylim)
    plt.xlabel("Contrast", fontsize=fontsize)
    plt.ylabel("Push counts", fontsize=fontsize)
    plt.title("Button push count - contrast", fontsize=fontsize)
    plt.legend(title="", fontsize=fontsize, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    
    
    
# Plotting calibration curve
def calibration_curve(ax, trials, linewidth=3, fontsize=15, HE_border=None, color_HE=color_HE):
    # Create a copy of the DataFrame
    df = trials[['Time_investment', 'Correct', 'Contrast']].copy()
    
    df['Time_investment'] = df['Time_investment'].floordiv(1)+0.5

    df['Correct'] = df['Correct'].astype(int)
    df['Contrast'] = df['Contrast'].abs()
    if HE_border is None:
        hue = "Contrast"
        palette = None
    else:
        df['Difficulty'] = df['Contrast'].apply(lambda x: "Hard" if x <= HE_border else "Easy")
        hue = "Difficulty"
        palette = color_HE
    
    # Plot using seaborn
    sns.lineplot(
        x="Time_investment", 
        y="Correct", 
        hue=hue,
        data=df, 
        estimator=np.mean,
        errorbar=('se', 1),
        linewidth=linewidth,
        palette=palette,
        ax=ax
    )
    utils.biofig(ax=ax, linewidth=linewidth, fontsize=fontsize)
    # Set axis labels and style
    plt.xlim([0, 15])
    plt.ylim([0, 1.1])
    plt.xlabel("Time investment (s)", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)
    plt.legend(title="", fontsize=fontsize, frameon=False)
    plt.title("Calibration curve", fontsize=fontsize)
    plt.grid(False)
    plt.tight_layout()
    


# Plotting vevaiometric curve 
def vevaiometric_curve(ax, trials, linewidth=3, fontsize=15, estimator = 'mean', ylim = [0, 16], color_PEC=color_PEC):
    # Create a copy of the DataFrame
    df = trials[['Contrast', 'Time_investment', 'Status']].copy()
    status_order = ['Correct', 'Probe', 'Error']
    df['Status'] = pd.Categorical(df['Status'], categories=status_order, ordered=True)
    df['Contrast'] = df['Contrast'].abs()

    # Plot using seaborn
    if estimator == 'mean':
        sns.lineplot(
            x="Contrast", 
            y="Time_investment", 
            hue="Status",
            data=df, 
            estimator=np.mean,
            errorbar=('se', 1), 
            linewidth=linewidth, 
            palette=color_PEC,
            ax=ax
        )
    elif estimator == 'median':
        sns.lineplot(
            x="Contrast", 
            y="Time_investment", 
            hue="Status",
            data=df, 
            estimator=np.median,
            errorbar=('pi', 50), 
            linewidth=linewidth, 
            palette=color_PEC,
            ax=ax
        )
    
    utils.biofig(ax=ax, linewidth=linewidth, fontsize=fontsize)
    # Set axis labels and style
    plt.xlim([0, 1.1])
    plt.ylim(ylim)
    plt.xlabel("|Contrast|", fontsize=fontsize)
    plt.ylabel("Time investment (s)", fontsize=fontsize)
    plt.title("Vevaiometric curve", fontsize=fontsize)
    plt.legend(title="", fontsize=fontsize, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    


# Plotting pshycometric functions with time investment
def psychometric_curve_condition_TI(ax, trials, linewidth=3, fontsize=15, border_time = 5, color_SL=color_SL):
    # Create a copy of the DataFrame
    df = trials[['Contrast', 'Choice', 'Time_investment']].copy()
    df['Choice'] = df['Choice'] == 'right'
    
    df['Time_investment'] = df['Time_investment'].apply(lambda x: "Long" if x > border_time else "Short")

    # Plot using seaborn
    sns.lineplot(
        data=df,
        x="Contrast", 
        y="Choice",
        hue = "Time_investment",
        estimator=np.mean,
        errorbar=('se', 1), 
        linewidth=linewidth,
        palette=color_SL
    )
    utils.biofig(ax=ax, linewidth=linewidth, fontsize=fontsize)
    # Set axis labels and style
    plt.xlim([-1.1, 1.1])
    plt.ylim([0, 1])
    plt.xlabel("Contrast", fontsize=fontsize)
    plt.ylabel("pRightward", fontsize=fontsize)
    plt.legend(title="", fontsize=fontsize, frameon=False)
    plt.title("Psychometric function", fontsize=fontsize)
    plt.grid(False)
    plt.tight_layout()
    


# Plotting pshycometric functions with time investment
def reaction_time_trials(ax, trials, linewidth=3, fontsize=15, mrksize=10, color_LR=color_LR):
    df = trials[['Trial', 'Stim', 'Choice', 'Correct', 'RT_left', 'RT_right']].copy()
    df["Correct"] = df["Correct"].apply(lambda x: "Correct" if x else "Error")
    df["Reaction_time"] = df.apply(lambda row: row["RT_left"] if row["Choice"] == "left" else row["RT_right"], axis=1)
    status_order = ['left', 'right']
    df['Stim'] = pd.Categorical(df['Stim'], categories=status_order, ordered=True)
    status_order = ['Correct', 'Error']
    df['Correct'] = pd.Categorical(df['Correct'], categories=status_order, ordered=True)

    # Plot parameters

    # Plot using seaborn
    sns.scatterplot(
        data=df,
        x="Reaction_time",
        y="Trial",
        hue="Stim",
        style="Correct",
        s = mrksize,
        palette=color_LR,
        ax = ax
    )
    utils.biofig(ax=ax, linewidth=linewidth, fontsize=fontsize)
    plt.xlim([0, 2])
    plt.ylim([0, df.shape[0]])
    plt.yticks(np.arange(0, df.shape[0], 50))
    plt.xlabel("Reaction time (s)", fontsize=fontsize)
    plt.ylabel("Trial", fontsize=fontsize)
    plt.title("Trial - Reaction time", fontsize=fontsize)
    plt.legend(title="", fontsize=fontsize, frameon=False)
    plt.grid(False)
    plt.tight_layout()



# Plotting correct rate
def correct_rate(ax, trials, linewidth=3, fontsize=15, window = 50, color_LR=color_LR):
    # Load target dataset
    df = trials[['Trial', 'Stim', 'Correct']].copy()
    df['cumulative_accuracy'] = df['Correct'].expanding().mean()
    df['rolling_accuracy'] = df['Correct'].rolling(window=20, min_periods=1).mean()
    df['rolling_accuracy_left'] = df[df['Stim'] == 'left']['Correct'].rolling(window=50, min_periods=1).mean()
    df['rolling_accuracy_right'] = df[df['Stim'] == 'right']['Correct'].rolling(window=50, min_periods=1).mean()
    df = df.iloc[1:].reset_index(drop=True)
    k_values = df['Correct'].cumsum().to_numpy().astype(int) 
    n_values = df["Trial"].to_numpy().astype(int) 

    p_values = []
    for k, n in zip(k_values, n_values):
        result = stats.binomtest(k, n, p=0.5, alternative='two-sided')
        if k/n>0.5:
            p_values.append(result.pvalue)
        else:
            p_values.append(1)
    df["p_value"] = p_values
    df["significant"] = np.where(df["p_value"] < 0.025, 1, 0)
    
    # Plot parameters
    color_L = color_LR['left']
    color_R = color_LR['right']
    
    # Plot using seaborn
    plt.fill_between(
        df["Trial"],
        df["significant"],
        label="Significant",
        color=(0.2, 0.3, 0.7, 0.1)
    )
    
    sns.lineplot(
        x=df['Trial'],
        y=df['rolling_accuracy_left'],
        label='Trial Rolling (Left)',
        color=color_L,
        linewidth=linewidth,
        alpha=0.7,
        ax=ax
    )
    sns.lineplot(
        x=df['Trial'],
        y=df['rolling_accuracy_right'],
        label='Trial Rolling (Right)',
        color=color_R,
        linewidth=linewidth,
        alpha=0.7,
        ax=ax
    )
    sns.lineplot(
        x=df['Trial'],
        y=df['rolling_accuracy'],
        label='Trial Rolling',
        color=(0.2, 0.2, 0.2),
        linewidth=linewidth,
        alpha=0.7,
        ax=ax
    )
    sns.lineplot(
        x=df['Trial'], 
        y=df['cumulative_accuracy'], 
        label='Cumulative', 
        color=(0, 0, 0), 
        linewidth=linewidth, 
        alpha=1,
        ax=ax
    )
    utils.biofig(ax=ax, linewidth=linewidth, fontsize=fontsize)
    plt.xlim([0, df.shape[0]])
    plt.ylim([0, 1])
    plt.xticks(np.arange(0, df.shape[0], 50))
    plt.xlabel("Trial", fontsize=fontsize)
    plt.ylabel("Correct rate", fontsize=fontsize)
    plt.title("Trial - Correct rate", fontsize=fontsize)
    plt.legend(title="", fontsize=fontsize, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    
    
    

def conditional_prob_prestim(ax1, trials, linewidth=3, fontsize=15, mrksize=10):
    # Copy necessary columns
    df = trials[["Stim", "Correct"]].copy()
    
    # Encode correctness
    df["Correct"] = df["Correct"].map({True: "Correct", False: "Error"})
    df["Prev_Stim"] = df["Stim"].shift(1)
    df["Prev_Correct"] = df["Correct"].shift(1)
    df = df.dropna().reset_index(drop=True)
    
    # Define all expected indices and columns for reindexing
    expected_index = pd.MultiIndex.from_product(
        [["left", "right"], ["Correct", "Error"]],
        names=["Prev_Stim", "Prev_Correct"]
    )
    expected_columns = ["left", "right"]

    # Compute counts, correct trials, and rate
    counts = df.groupby(["Prev_Stim", "Prev_Correct", "Stim"]).size().unstack(fill_value=0)
    correct = df[df["Correct"] == "Correct"].groupby(["Prev_Stim", "Prev_Correct", "Stim"]).size().unstack(fill_value=0)

    # Reindex to ensure 4x2 structure
    counts = counts.reindex(index=expected_index, columns=expected_columns, fill_value=0)
    correct = correct.reindex(index=expected_index, columns=expected_columns, fill_value=0)

    # Compute correct rate
    rate = correct.div(counts).fillna(0)

    # Flatten each DataFrame in the specified order (LC, RE, LE, RC)
    def flatten_in_order(df):
        LC = df.iloc[0:1]
        RE = df.iloc[3:4]
        LE = df.iloc[1:2]
        RC = df.iloc[2:3]
        return pd.concat([LC, RE, LE, RC]).T.to_numpy().reshape(-1, 1)

    y1 = flatten_in_order(counts)
    y2 = flatten_in_order(correct)
    y3 = flatten_in_order(rate)

    # Consistency check (optional)
    assert y1.shape[0] == 8
    assert y2.shape[0] == 8
    assert y3.shape[0] == 8

    # Labels aligned with the y-values
    labels = ["Lo>L", "Rx>L", "Ro>L", "Ro>L", "Lo>R", "Rx>R", "Ro>R", "Ro>R"]

    # Bar plots for counts and correct
    sns.barplot(
        x=range(len(labels)),
        y=y1.flatten(),
        linewidth=linewidth,
        color=(0.5, 0.7, 0.7),
        ax=ax1
    )
    sns.barplot(
        x=range(len(labels)),
        y=y2.flatten(),
        linewidth=linewidth,
        color=(0.1, 0.3, 0.3),
        ax=ax1
    )

    # Configure ax1
    ax1.set_ylim(0, y1.max() + 10 if y1.size > 0 else 300)
    ax1.set_ylabel("Trials", color=(0.1, 0.5, 0.5), fontsize=fontsize)
    ax1.tick_params(axis='y', labelcolor=(0.1, 0.5, 0.5), labelsize=fontsize)
    ax1.spines["left"].set_color((0.1, 0.5, 0.5))
    ax1.spines['left'].set_linewidth(linewidth)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlabel("Condition", fontsize=fontsize)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels=labels, fontsize=fontsize, rotation=45)
    ax1.set_title("Conditional probability", fontsize=fontsize)

    # Line plot for correct rate on right axis
    ax2 = ax1.twinx()
    sns.lineplot(
        x=range(len(y3)),
        y=y3.flatten(),
        linewidth=linewidth,
        marker="o",
        markersize=mrksize,
        color=(0.5, 0.2, 0.2),
        ax=ax2
    )
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Correct rate", color=(0.5, 0.2, 0.2), fontsize=fontsize)
    ax2.tick_params(axis='y', labelcolor=(0.5, 0.2, 0.2), labelsize=fontsize)
    ax2.spines["right"].set_color((0.5, 0.2, 0.2))
    ax2.spines['right'].set_linewidth(linewidth)
    ax2.spines['top'].set_visible(False)

    plt.grid(False)
    plt.tight_layout()

    
    
    
# Plotting correlation between time investment and selection time
def corr_TI_ST(ax, trials, linewidth=3, fontsize=15, mrksize=10, color_PEC=color_PEC, outliner_cut_method = ("percentile", 95)):

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

    # Set the color
    color_P = color_PEC["Probe"]
    color_E = color_PEC["Error"]

    # Plot seaborn
    sns.regplot(
        data=df_probe,
        x="Selection_time",
        y="Time_investment",
        color=color_P,
        marker="o",
        scatter_kws={"s": mrksize, "alpha": 0.5},
        ax=ax,
    )
    sns.regplot(
        data=df_error,
        x="Selection_time",
        y="Time_investment",
        color=color_E,
        marker="x",
        scatter_kws={"s": mrksize, "alpha": 0.5},
        ax=ax,
    )
    utils.biofig(linewidth=linewidth, fontsize=fontsize)
    plt.xlim(0, 8)
    plt.ylim(0, 20)
    plt.xlabel("Selection time (s)")
    plt.ylabel("Time investment (s)")
    plt.text(1, 12, f"Probe: ρ={rho_probe:.2f}, p={p_probe:.3f}", fontsize=fontsize, color=(0.2, 0.3, 0.7))
    plt.text(1, 10, f"Error: ρ={rho_error:.2f}, p={p_error:.3f}", fontsize=fontsize, color=(0.7, 0.3, 0.2))
    plt.title("Correlation between TI and ST", fontsize=fontsize)
    plt.grid(False)
    plt.tight_layout()



# Plotting chronometric function
def chronometric_function(ax, trials, linewidth=3, fontsize=15, mrksize=10):

    df = trials[["Contrast", "Status", "Time_investment"]].copy()
    df = df[df["Status"] != "Correct"]
    abscont = df['Contrast'].abs() 
    ce = df["Status"].apply(lambda x: 1 if x == "Probe" else -1)
    evidence = abscont * ce
    df["Evidence"]  = evidence
    df = df.dropna()
    
    if evidence.unique().size > 2:
        rho, p_value = stats.spearmanr(df["Evidence"], df["Time_investment"])
    else:
        rho = 0
        p_value = 1
    
    # Plot seaborn
    sns.regplot(
        data=df,
        x="Evidence",
        y="Time_investment",
        marker="o",
        scatter_kws={"s": mrksize, "alpha": 0.5},
        ax=ax,
    )
    utils.biofig(linewidth=linewidth, fontsize=fontsize)
    plt.xlim(-1.1, 1.1)
    plt.ylim(0, 15)
    plt.xlabel("|Δcontrast| x correct")
    plt.ylabel("Time investment (s)")
    plt.text(-1, 14, f"ρ = {rho:.2f}, p = {p_value:.3f}", fontsize=fontsize)
    plt.title("Chronometric", fontsize=fontsize)
    plt.grid(False)
    plt.tight_layout()
    



def push_count_heatmap(ax, trials, linewidth=3, fontsize=15):
    df = trials.copy()
    df["Push_count_choice"] = df.apply(lambda row: row["PC_left"] if row["Choice"] == "left" else row["PC_right"], axis=1)
    df["Push_count_init"]   = df.apply(lambda row: row["IPC_left"] if row["Choice"] == "right" else row["IPC_right"], axis=1)

    matrix = pd.crosstab(df["Push_count_init"], df["Push_count_choice"], normalize=True)
    
    sns.heatmap(
        matrix,
        cmap="Blues",
        ax=ax,
    )
    
    utils.biofig(linewidth=linewidth, fontsize=fontsize)
    plt.xlim([0, 6])
    plt.ylim([0, 6])
    plt.xlabel("Choice", fontsize=fontsize)
    plt.ylabel("Initialization", fontsize=fontsize)
    plt.title("Push count", fontsize=fontsize)
    plt.grid(False)
    #plt.tight_layout()




# ============================== Entry Point ==============================
if __name__ == "__main__":
    root_folder = 'Z:/Data'
    output_dir = 'Z:\Figure\Behavior\session/'
    target_subject = ['RSS023', 'RSS025', 'RSS026', 'RSS027', 'RSS030', 'RSS033', 'RSS038', 'RSS039', 'RSS040', 'RSS041']
    plot_auto(root_folder, output_dir, target_subject)