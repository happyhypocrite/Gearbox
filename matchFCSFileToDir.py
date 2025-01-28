import os
import pandas as pd

csv_file_path = 'C:\\Users\\mfbx2rdb\\OneDrive - The University of Manchester\\PDRA\\Sequencing\\Py scripts\\Projects\\ImmAcc\\Gearbox\\StrokeIMPaCT_SmartTube_V2\\fcs_metadata_22012025.csv'
directory_path = 'C:\\Users\\mfbx2rdb\\OneDrive - The University of Manchester\\PDRA\\Sequencing\\Py scripts\\Projects\\ImmAcc\\Gearbox\\StrokeIMPaCT_SmartTube_V2\\StrokeIMPaCT_SmartTube123_V2'

df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
csv_filenames = df['Filename'].tolist()

directory_files = os.listdir(directory_path)
common_files = [file for file in csv_filenames if file in directory_files]
missing_in_dir = [file for file in csv_filenames if file not in directory_files]
extra_in_dir = [file for file in directory_files if file not in csv_filenames]

print("Common files:", common_files)
print("Missing in directory:", missing_in_dir)
print("Not in CSV:", extra_in_dir)