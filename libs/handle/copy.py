import pandas as pd
import libs.utils.utils as utils



# Main script
if __name__ == "__main__":
    file_path = r'Z:\Handle_data\mainDirs_for_IoC.csv'
    IoC_dir_Info = pd.read_csv(file_path)
    print(IoC_dir_Info)
    
    output_path = r'E:\Z'
    folders = ['procTeensy', 'spikeinterface', 'processed-data']
    
    
    for i_session in range(2, len(IoC_dir_Info)):
        session_dir = IoC_dir_Info.iloc[i_session]['Session_dir']
        utils.printlg(f"Copying session {i_session} from {session_dir} to {output_path}")
        source_path, dest_base = utils.copy(
            source_path=session_dir,
            output_path=output_path,
            path_mode='full',
            folders=folders)