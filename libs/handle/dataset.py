import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pickle
import libs.utils.utils as utils
import libs.prep_behavior.procTeensy_to_pkl as pt2pkl
import libs.prep_behavior.generate_trials as gen_trials
import spikeinterface as si



class Session:
    def __init__(self, session_dir):
        
        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"Session directory {session_dir} does not exist.")
        
        self.session_dir = session_dir
        self.metaFile = self.load_metaFile()
        self.trials = self.load_trials()
    
    
    # Helper functions
    def _load_pickle(self, relative_path):
        path = os.path.join(self.session_dir, relative_path)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return None 
        with open(path, 'rb') as f:
            return pickle.load(f)


    # Load dataset
    def load_metaFile(self):
        return self._load_pickle(os.path.join('procTeensy', 'metaFile.pkl'))
    
    def load_procTeensy(self):
        return self._load_pickle(os.path.join('procTeensy', 'procTeensy.pkl'))
        
    def load_event(self):
        return self._load_pickle(os.path.join('procTeensy', 'event.pkl'))

    def load_trials(self):
        return self._load_pickle(os.path.join('procTeensy', 'trials.pkl'))
        
    def load_sorting_analyzer(self, i_imec):
        imec_folder = 'imec'+str(i_imec)
        base_dir = os.path.join(self.session_dir, 'spikeinterface', imec_folder)
        path = os.path.join(base_dir, 'sorting_analyzer')
        sorting_analyzer = si.load(path)
        return sorting_analyzer, base_dir
    
    def load_analyzer_clean(self, i_imec):
        imec_folder = 'imec'+str(i_imec)
        base_dir = os.path.join(self.session_dir, 'spikeinterface', imec_folder)
        path = os.path.join(base_dir, 'analyzer_clean')
        analyzer_clean = si.load(path)
        return analyzer_clean, base_dir
    
    
    def load_procTeensy_to_imec(self, i_imec):
        imec_folder = 'imec'+str(i_imec)
        processed_data = os.path.join(self.session_dir, 'processed-data', imec_folder)
        file_name = 'procTeensy_to_'+imec_folder+'.npy'
        path = os.path.join(processed_data, file_name)
        procTeensy_to_imec = np.load(path)
        return procTeensy_to_imec


    # Load dataset and add to session
    def add_metaFile(self):
        self.meta_file = self.load_metaFile()
        return self
    
    def add_procTeensy(self):
        self.procTeensy = self.load_procTeensy()
        return self

    def add_event(self):
        self.event = self.load_event()
        return self
    
    def add_trials(self):
        self.trials = self.load_trials()
        return self
    
    def add_sorting_analyzer(self, i_imec):
        sorting_analyzer, base_dir = self.load_sorting_analyzer(i_imec=i_imec)
        self.sorting_analyzer = sorting_analyzer
        self.sorting_analyzer_base_dir = base_dir
        return self
    
    def add_analyzer_clean(self, i_imec):
        analyzer_clean, base_dir = self.load_analyzer_clean(i_imec=i_imec)
        self.analyzer_clean = analyzer_clean
        self.analyzer_clean_base_dir = base_dir
        return self
    
    def add_procTeensy_to_imec(self, i_imec):
        self.procTeensy_to_imec = self.load_procTeensy_to_imec(i_imec=i_imec)
        return self
    
    
    # Recalculate
    def recalc_metaFile(self):
        procTeensy_dir = os.path.join(self.session_dir, 'procTeensy')
        self.meta_file = pt2pkl.metafile(procTeensy_dir, procTeensy_dir, save=True)    
        utils.printlg('Recalculated metaFile.pkl')
        return self.meta_file
    
    def recalc_trials(self):
        trials = gen_trials.generate_trials(self.metaFile, self.load_event())
        trials.to_pickle(path = os.path.join(self.session_dir, 'procTeensy', 'trials.pkl'))
        self.trials = trials
        utils.printlg('Recalculated trials.pkl')
        return self.trials
    