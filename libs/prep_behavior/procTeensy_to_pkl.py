import os
import numpy as np
import pandas as pd
from scipy import stats
import csv
import pickle
import libs.utils.utils as utils
import matplotlib.pyplot as plt
import libs.prep_behavior.generate_procTeensy_and_event as gen_prcTnsy_event
import libs.prep_behavior.generate_trials as gen_trials
import libs.visualization.behavior_single_session as plt_behv_snglssn


def procTeensy_to_pkl_auto(root_folder=None, output_dir=None, target_subjects=None, pltfig=False, confirm_pkls=True):
    # Data folder path
    if root_folder is None:
        utils.printlg('Error | root_folder is None')
    if output_dir is None:
        utils.printlg('Error | output_folder is None')
    if target_subjects is None:
        utils.printlg('Error | target_subjects is None')
        
    
    subject_folders = utils.get_subfolders(root_folder)

    subject_folders = subject_folders[subject_folders['folder_name'].isin(target_subjects)]
    subject_folders = subject_folders.sort_values(by="folder_name", ascending=True).reset_index(drop=True)
    sz = len(subject_folders)

    # Loop Subject
    for i_subject in range(0, len(subject_folders)):
        session_folders = utils.get_subfolders(subject_folders.iloc[i_subject] ['absolute_path'])
        #print(session_folders)
        
        # Loop session
        for i_session in range(0, len(session_folders)):

            session_folder = session_folders.loc[i_session]['absolute_path']
            input_dir = os.path.join(session_folder, 'procTeensy')

            os.chdir(input_dir)
            
            files_to_check = ['metaFile.pkl', 'procTeensy.pkl', 'event.pkl', 'trials.pkl']
            existing_files = [file for file in files_to_check if os.path.exists(os.path.join(input_dir, file))]
            
            if (len(existing_files) == len(files_to_check))&confirm_pkls:
                print("File dir:  "+ os.getcwd()+"  |  Already exists")
            else:
                print("File dir:  "+ os.getcwd())
                # Loading and processing data
                # Input dir = output dir
                meta_file, procTeensy, event, trials = procTeensy_to_pkl(input_dir, input_dir)
                del procTeensy, event
            
                # Plotting
                if pltfig:
                    trials = trials.dropna(subset=['Trial'])
                    plt_behv_snglssn.summary(meta_file, trials, output_dir, savefig=True, pltshow=False)
                    plt_behv_snglssn.summary(meta_file, trials, input_dir, savefig=True, pltshow=False)
                    
                    
                    
def procTeensy_to_pkl(input_dir, output_dir):
    
    utils.printlg("File dir: "+input_dir)

    meta_file         = metafile(input_dir, output_dir, save=True)
    tnsy              = teensy(input_dir, output_dir, save=False, pltshow=False)
    procTeensy, event = procTeensyAndEvent(input_dir, output_dir, meta_file, tnsy, pltshow=False)
    
    procTeensy.to_pickle(os.path.join(output_dir, "procTeensy.pkl"))
    event.to_pickle(os.path.join(output_dir, "event.pkl"))

    trials = gen_trials.generate_trials(meta_file, event)
    trials.to_pickle(os.path.join(output_dir, "trials.pkl"))
    
    return meta_file, procTeensy, event, trials
    



def metafile(input_dir, output_dir, save=False):
    # Loading metafile
    path = os.path.join(input_dir, "metaFile.csv") 
    TF = os.path.exists(path)
    
    if TF == False:
        print("File not found")
        return None
        
    df = pd.read_csv(path, encoding="utf-8", skip_blank_lines=True)

    header = pd.DataFrame({"Name": df.columns[::2], "Value": df.columns[1::2]})
    time = df.iloc[0:3, 0:2].rename(columns={df.columns[0]: "Name", df.columns[1]: "Value"})

    trial = df.iloc[3:4, 1:df.shape[1]].T.reset_index(drop=True)
    trial.columns = ["Count"]
    trial["State"] = range(1, trial.shape[0] + 1)
    trial = trial[["State", "Count"]]

    correct_rate = df.iloc[4:5, 1:4].T.reset_index(drop=True)
    correct_rate.columns = ["Correct_rate"]
    correct_rate["Name"] = ["All", "Left", "Right"]
    correct_rate = correct_rate[["Name", "Correct_rate"]]

    contrast = df.iloc[5:7, 1:df.shape[1]].T.reset_index(drop=True)
    contrast = contrast.rename(columns={contrast.columns[0]: "Low_contrast", contrast.columns[1]: "High_contrast"})
    contrast.dropna(inplace=True)
    
    body_weight = pd.DataFrame({
        'Pre': [df.iloc[7, 1]],
        'Post': [df.iloc[8, 1]]
    })
    body_weight = body_weight.astype(float)

    meta_file = {
        "Header": header,
        "Time": time,
        "Trial": trial,
        "Correct_rate": correct_rate,
        "Contrast": contrast,
        "Body_weight": body_weight
    }
    if save:
        with open(os.path.join(output_dir, "metaFile.pkl"), "wb") as f:
            pickle.dump(meta_file, f)

    utils.printlg("Done | metaFile.csv loaded")
    return meta_file



def cursorPos(input_dir, output_dir, save=False):
    # Loading cursorPos.csv
    path = os.path.join(input_dir, 'cursorPos.csv')
    cursorPos = []
    with open(path, encoding = 'utf-8') as f:
        reader = csv.reader(f)
        # Read from line 3
        for _ in range(2): 
            next(reader)
        # Read from column 5
        for row in reader:
            data = row[4:504]
            cursorPos.append(data)
    cursorPos = [item for sublist in cursorPos for item in sublist]
    cursorPos = np.array(cursorPos)
    
    if save:
        with open(os.path.join(output_dir, "cursorPos.pkl"), "wb") as f:
            pickle.dump(cursorPos, f)
    
    utils.printlg("Done | cursorPos.csv loaded")
    return cursorPos



def timeStamp(input_dir, output_dir, save=False):
    # Loading timeStamp.csv
    path = os.path.join(input_dir, 'timeStamp.csv')
    timeStamp = []
    with open(path, encoding = 'utf-8') as f:
        reader = csv.reader(f)
        # Read from line 3
        for _ in range(2): 
            next(reader)
        # Read from column 5
        for row in reader:
            data = row[4:504]
            timeStamp.append(data)
    timeStamp = [item for sublist in timeStamp for item in sublist]
    timeStamp = np.array(timeStamp)
    
    if save:
        with open(os.path.join(output_dir, "timeStamp.pkl"), "wb") as f:
            pickle.dump(timeStamp, f)
    
    utils.printlg("Done | timeStamp.csv loaded")
    return timeStamp



def teensy(input_dir, output_dir, save=False, pltshow=False):
    # Loading teensy.csv
    path = os.path.join(input_dir, 'teensy.csv')
    teensy = []
    with open(path, encoding = 'utf-8') as f:
        reader = csv.reader(f)
        # Read from line 3
        for _ in range(2): 
            next(reader)
        # Read from column 5
        for row in reader:
            data = row[4:504]
            teensy.append(data)
    
    # Plot row lengths
    row_lengths = [len(sublist) for sublist in teensy]
    save_path = os.path.join(output_dir, "teensy_row_lengths.jpg")
    plt.figure()
    plt.plot(row_lengths)
    plt.title("Length of each row in teensy.csv")
    plt.xlabel("Row")
    plt.ylabel("Length")
    plt.savefig(save_path, format="jpg", dpi=300)
    if pltshow:
        plt.show()
    else:
        plt.close()
    
    # Convert Nx1 array
    teensy = [item for sublist in teensy for item in sublist]
    teensy = np.array(teensy)

    if save:
        with open(os.path.join(output_dir, "teensy.pkl"), "wb") as f:
            pickle.dump(teensy, f)
    
    utils.printlg("Done | teensy.csv loaded")
    return teensy




def procTeensyAndEvent(input_dir, output_dir, meta_file, teensy, save=True, pltshow=False):

    header = meta_file['Header']
    program_name = header.loc[header["Name"] == "programName", "Value"] 
    
    utils.printlg("program_name = "+str(program_name.values[0]))
    
    if program_name.values[0] == 'dm2afc_gc_v001':
        procTeensy, event = gen_prcTnsy_event.dm2afc_gaborContrast_v001(
            input_dir=input_dir, 
            output_dir=output_dir, 
            teensy=teensy, 
            trial_start_state_num=1,
            reg_trial=0, 
            pltshow=pltshow
        )
        
    elif program_name.values[0] == 'dm2afc_cousor_v001':
        procTeensy, event = gen_prcTnsy_event.illusion_of_control(
            input_dir=input_dir, 
            output_dir=output_dir, 
            teensy=teensy,
            trial_start_state_num=1,
            reg_trial=0,
            pltshow=pltshow
        )
    
    elif program_name.values[0] == 'dm2afc_cursor_v001':
        procTeensy, event = gen_prcTnsy_event.illusion_of_control(
            input_dir=input_dir, 
            output_dir=output_dir, 
            teensy=teensy,
            trial_start_state_num=1,
            reg_trial=0,
            pltshow=pltshow
        )
        
    elif program_name.values[0] == 'dm2afc_illusion_of_control_v001':
        procTeensy, event = gen_prcTnsy_event.illusion_of_control(
            input_dir=input_dir, 
            output_dir=output_dir, 
            teensy=teensy,
            trial_start_state_num=2,
            reg_trial=0,
            pltshow=pltshow
        )
        
    elif program_name.values[0] == 'dm2afc_illusion_of_control_v002':
        procTeensy, event = gen_prcTnsy_event.illusion_of_control(
            input_dir=input_dir, 
            output_dir=output_dir, 
            teensy=teensy,
            trial_start_state_num=2,
            reg_trial=0,
            pltshow=pltshow
        )

    
    else:
        print("Error")
    
    if save:
        procTeensy.to_pickle(os.path.join(output_dir, "procTeensy.pkl"))
        event.to_pickle(os.path.join(output_dir, "event.pkl"))
    
    return procTeensy, event



# Main script
if __name__ == "__main__":
    # Auto procTeency to .pkl for all subjects
    root_folder = r'Z:/Data'
    output_dir = r'Z:\Figure\Behavior\session/'
    #target_subjects = ['RSS023', 'RSS025', 'RSS026', 'RSS027', 'RSS030', 'RSS033', 'RSS036', 'RSS038', 'RSS039', 'RSS040']
    target_subjects = ['RSS041', 'RSS044', 'RSS045', 'RSS046']
    procTeensy_to_pkl_auto(
        root_folder=root_folder,
        output_dir=output_dir,
        target_subjects=target_subjects,
        pltfig=True,
        confirm_pkls=True) 
    
    # Generate pooled trials
    print("Generating pooled trials")
    root_dir = r'Z:/Data'
    save_path = r'Z:/Pool/Behavior/trials_pooled.pkl'
    target_subjects = ['RSS023', 'RSS025', 'RSS026', 'RSS027', 'RSS030', 'RSS033', 'RSS036', 'RSS038', 'RSS039', 'RSS040', 'RSS041', 'RSS044', 'RSS045', 'RSS046']
    trials_pooled = gen_trials.load_pooled_trials(root_dir=root_dir, target_subjects=target_subjects, save_path=save_path, check=False)

