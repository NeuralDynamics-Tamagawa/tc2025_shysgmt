import os
import numpy as np
import pandas as pd
from scipy import stats
import csv
import pickle
import libs.utils.utils as utils
import libs.handle.dataset as handle_dataset
import spikeinterface.extractors as se
import spikeinterface as si
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as scur



def generate_sorting_analyzer(session_dir, i_imec=None, extension=False):
    
    job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
    
    utils.printlg("Gen. sorting_analyzer: Session directory: "+session_dir)
    
    # Directories
    neuropixels_dir    = os.path.join(session_dir, 'neuropixels')
    kilosort4_dir      = os.path.join(session_dir, 'kilosort4')
    processed_data_dir = os.path.join(session_dir, 'processed-data')

    neuropixels_subdirs     = utils.get_subfolders(neuropixels_dir) 
    kilosort4_subdirs       = utils.get_subfolders(kilosort4_dir)
    processed_data_subdirs  = utils.get_subfolders(processed_data_dir)

    if i_imec is None:
        iter = range(0, len(neuropixels_subdirs))
    else:
        iter = range(i_imec, i_imec+1)
    
    for i_imec in iter:
        utils.printlg("Gen. sorting_analyzer: imec"+str(i_imec))
        # Generate folders
        imec_folder = 'imec'+str(i_imec)

        # Load dataset directories
        neuropixels_dir     = neuropixels_subdirs.iloc[i_imec]['absolute_path']    
        kilosort4_dir       = kilosort4_subdirs.iloc[i_imec]['absolute_path']
        processed_data_dir  = processed_data_subdirs.iloc[i_imec]['absolute_path']

        # output directories
        spikeinterface_dir = os.path.join(session_dir, "spikeinterface")
        os.makedirs(spikeinterface_dir, exist_ok=True)
        
        

        # Load recording and sorting
        utils.printlg("Gen. sorting_analyzer: Loading recording and sorting")
        recording = se.read_spikeglx(neuropixels_dir, stream_name='imec0.ap')
        sorting = se.read_kilosort(kilosort4_dir)
        analyzer = si.create_sorting_analyzer(recording=recording, sorting=sorting)
        
        if extension is True:
            
            analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
            analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
            analyzer.compute("templates", operators=["average", "median", "std"])
            analyzer.compute(["noise_levels"]) 
            analyzer.compute('spike_amplitudes', **job_kwargs)
            analyzer.compute("correlograms")
            analyzer.compute("unit_locations")
            analyzer.compute("template_similarity")
            analyzer.compute("template_metrics", include_multi_channel_metrics=True, save=True)
        
        

        # Save analyzer to disk (format='binary_folder')
        utils.printlg("Gen. sorting_analyzer: Saving sorting analyzer")
        spikeinterface_imec_dir = os.path.join(spikeinterface_dir, imec_folder)
        os.makedirs(spikeinterface_imec_dir, exist_ok=True)
        save_dir = os.path.join(spikeinterface_imec_dir, 'sorting_analyzer')
        analyzer.save_as(folder=save_dir, format='binary_folder')
        
        utils.printlg("Gen. sorting_analyzer: Saved sorting_analyzer to disk ("+save_dir+")")
        


def generate_extensions(session_dir, i_imec=None):
    
    job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
    
    utils.printlg("Spikeinterface (Compute extensions): Session directory: "+session_dir)
    
    spikeinterface_dir     = os.path.join(session_dir, 'spikeinterface')
    spikeinterface_subdirs = utils.get_subfolders(spikeinterface_dir)
    
    utils.printlg("Spikeinterface subfirs")
    utils.printlg(spikeinterface_subdirs)
    
    if i_imec is None:
        iter = range(0, len(spikeinterface_subdirs))
    else:
        iter = range(i_imec, i_imec+1)
        
    session = handle_dataset.Session(session_dir)
    
    for i_imec in iter:
        utils.printlg("Gen. extensions: imec"+str(i_imec))
        utils.printlg("  loading analyzer"+str(i_imec))
        session.add_sorting_analyzer(i_imec=i_imec)
        analyzer = session.sorting_analyzer
        
        utils.printlg("  compute extension"+str(i_imec))
        
        # Compute extensions
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500, save=True)
        analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, save=True)
        analyzer.compute("templates", operators=["average", "median", "std"], save=True)
        analyzer.compute(["noise_levels"], save=True) 
        analyzer.compute('spike_amplitudes', **job_kwargs, save=True)
        analyzer.compute("correlograms", save=True)
        analyzer.compute("unit_locations", save=True)
        analyzer.compute("template_similarity", save=True)
        analyzer.compute("template_metrics", include_multi_channel_metrics=True, save=True)
        




def generate_quality_metrics(session_dir, i_imec=None):
    utils.printlg("Spikeinterface (Compute quality_metrics): Session directory: "+session_dir)
    
    metric_names=['firing_rate', 
                  'presence_ratio', 
                  'snr', 
                  'isi_violation', 
                  'rp_violation', 
                  'sliding_rp_violation', 
                  'amplitude_cutoff', 
                  'amplitude_median', 
                  'amplitude_cv']
    
    spikeinterface_dir     = os.path.join(session_dir, 'spikeinterface')
    spikeinterface_subdirs = utils.get_subfolders(spikeinterface_dir)
    
    utils.printlg("Spikeinterface subfirs")
    utils.printlg(spikeinterface_subdirs)
    
    if i_imec is None:
        iter = range(0, len(spikeinterface_subdirs))
    else:
        iter = range(i_imec, i_imec+1)
        
    session = handle_dataset.Session(session_dir)
    
    for i_imec in iter:
        utils.printlg("Gen. quality metrics: imec"+str(i_imec))
        session.add_sorting_analyzer(i_imec=i_imec)
        analyzer = session.sorting_analyzer
        
        utils.printlg("  curation: imec"+str(i_imec))
        sorting = analyzer.sorting
        recording = analyzer.recording
        sorting = scur.remove_excess_spikes(sorting, recording)
        analyzer.sorting = sorting
        
        utils.printlg("  compute quality_metrics"+str(i_imec))
        sqm.compute_quality_metrics(analyzer, metric_names=metric_names, save=True)
        


def generate_clean_analyzer_and_report(session_dir, i_imec=None, our_query=None):
    utils.printlg("Spikeinterface (Clean analyzer and report): Session directory: "+session_dir)
    
    if our_query is None:
        amplitude_cutoff_thresh = 0.1
        isi_violations_ratio_thresh = 0.2
        presence_ratio_thresh = 0.9
        our_query = f"(amplitude_cutoff < {amplitude_cutoff_thresh}) & (isi_violations_ratio < {isi_violations_ratio_thresh}) & (presence_ratio > {presence_ratio_thresh})"
    utils.printlg("our_query: "+our_query)
    
    spikeinterface_dir     = os.path.join(session_dir, 'spikeinterface')
    spikeinterface_subdirs = utils.get_subfolders(spikeinterface_dir)
    
    utils.printlg("Spikeinterface subfirs")
    utils.printlg(spikeinterface_subdirs)
    
    if i_imec is None:
        iter = range(0, len(spikeinterface_subdirs))
    else:
        iter = range(i_imec, i_imec+1)
        
    session = handle_dataset.Session(session_dir)
    
    for i_imec in iter:
        utils.printlg("Gen. clean_analyzer and report: imec"+str(i_imec))
        
        utils.printlg("   Loading dataset: imec"+str(i_imec))
        session.add_sorting_analyzer(i_imec=i_imec)
        analyzer = session.sorting_analyzer
        base_dir = session.sorting_analyzer_base_dir
        
        sorting = analyzer.sorting
        recording = analyzer.recording
        extensions = analyzer.extensions
        
        metrics = extensions['quality_metrics'].get_data()
        
        utils.printlg("   Extract clean unit idx: imec"+str(i_imec))
        keep_units = metrics.query(our_query)
        keep_unit_ids = keep_units.index.values
        
        
        utils.printlg("  Curation: imec"+str(i_imec))
        sorting = scur.remove_excess_spikes(sorting, recording)
        analyzer.sorting = sorting
        
        utils.printlg("  Generate analyzer_clean: imec"+str(i_imec))
        save_dir = os.path.join(base_dir, 'analyzer_clean')
        analyzer_clean = analyzer.select_units(keep_unit_ids, folder=save_dir, format='binary_folder')
        
        utils.printlg("  Curation: imec"+str(i_imec))
        sorting = analyzer_clean.sorting
        sorting = scur.remove_excess_spikes(sorting, recording)
        analyzer_clean.sorting = sorting
        
        utils.printlg("  Generate report: imec"+str(i_imec))
        save_dir = os.path.join(base_dir, 'report')
        si.export_report(analyzer_clean, save_dir, format='png')
        



if __name__ == "__main__":
    
    session_dir = r'Z:\Data\RSS030\RSS030_240820_152308'
    i_imec = 1
    generate_sorting_analyzer(session_dir, i_imec=i_imec, extension=False)
    generate_extensions(session_dir, i_imec=i_imec)
    generate_quality_metrics(session_dir, i_imec=i_imec)
    
    '''
    file_path = r'Z:\Handle_data\mainDirs_for_IoC.csv'
    IoC_dir_Info = pd.read_csv(file_path)
    #IoC_dir_Info = IoC_dir_Info.drop(index=range(0, 1)).reset_index(drop=True)
    print(IoC_dir_Info)
    
    for i_session in range(20, len(IoC_dir_Info)):
        session_dir = IoC_dir_Info.iloc[i_session]['Session_dir']
        #generate_clean_analyzer_and_report(session_dir, i_imec=None)
    
    print("Done!")
    '''