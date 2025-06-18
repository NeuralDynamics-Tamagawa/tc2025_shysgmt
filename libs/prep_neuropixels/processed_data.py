import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import libs.utils.utils as utils
import libs.handle.dataset as handle_dataset 



def save_qm_tocsv_pdfolder(session_dir):
    
    subfolders = utils.get_subfolders(os.path.join(session_dir, 'processed-data'))
    
    for i_imec in range(0, len(subfolders)):
        # Load the session and add analyzers
        utils.printlg('Loading session and analyzers for imec'+str(i_imec))
        session = handle_dataset.Session(session_dir)
        session.add_analyzer_clean(i_imec=i_imec)
        session.add_sorting_analyzer(i_imec=i_imec)
        
        # Calc template metrics
        utils.printlg('Calc template metrics'+str(i_imec))
        session.sorting_analyzer.compute(input="template_metrics", include_multi_channel_metrics=True, save=True)
        session.analyzer_clean.compute(input="template_metrics", include_multi_channel_metrics=True, save=True)
        
        # Load KSLabel from sorting 
        KSLabel = session.sorting_analyzer.sorting.get_property('KSLabel')

        # Load quality metrics and clean metrics
        quality_metrics = session.sorting_analyzer.extensions['quality_metrics'].get_data()
        qm_clean = session.analyzer_clean.extensions['quality_metrics'].get_data()

        # Add KSLabel and in_clean to quality_metrics DataFrame
        quality_metrics['KSLabel'] = KSLabel
        quality_metrics['in_clean'] = quality_metrics.index.isin(qm_clean.index)

        # Save the updated quality metrics DataFrame
        save_path = os.path.join(subfolders.iloc[i_imec]['absolute_path'], 'quality_metrics.csv')
        quality_metrics.to_csv(save_path, index=True)
        utils.printlg('Saved quality metrics with KSLabel and in_clean for imec'+str(i_imec))




if __name__ == "__main__":
    file_path = r'Z:\Handle_data\mainDirs_for_IoC.csv'
    IoC_dir_Info = pd.read_csv(file_path)
    print(IoC_dir_Info)
    
    log_data = pd.DataFrame(columns=['Session_dir'])
     
    for i_session in range(0, len(IoC_dir_Info)):
        session_dir = IoC_dir_Info.iloc[i_session]['Session_dir']
        save_qm_tocsv_pdfolder(session_dir)
        print('Processed session:', session_dir)
        
        new_row = pd.DataFrame([{'Session_dir': session_dir}])
        log_data = pd.concat([log_data, new_row], ignore_index=True)
        
    save_dir = 'Z:\misc'
    log_data.to_csv(os.path.join(save_dir, 'log.csv'), index=False)