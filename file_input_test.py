import os
import pandas as pd
import fcsparser

csv_file_path = input(str('Enter path to .csv of metadata: '))
directory_path = input(str('Enter path to directory containing .fcs files: '))
directory_files = os.listdir(directory_path)

def test_file_size():
    file_size_dict = {}
    for file in directory_files:
        if file.endswith('.fcs'):
            fcs_file_path = os.path.join(directory_path, file)
            fcs_file_size = os.path.getsize(fcs_file_path)
            file_size_dict[fcs_file_path] = fcs_file_size
    fcs_file_errored = [key for key, size in file_size_dict.items() if size <= 1]
    if fcs_file_errored:
        print("FILE SUBMISSION FAILED")
        print("The following file(s) do not meet size criteria: ", fcs_file_errored)
    else:
        print("FILE SUBMISSION SUCCEEDED SIZE TEST")

def test_file_vs_meta():
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
    csv_filenames = df['Filename'].tolist()
    
    common_files = [file for file in csv_filenames if file in directory_files]
    missing_in_dir = [file for file in csv_filenames if file not in directory_files]
    extra_in_dir = [file for file in directory_files if file not in csv_filenames]

    if missing_in_dir or extra_in_dir is not None:
        print("FILE SUBMISSION FAILED")
        print("Missing in directory:", missing_in_dir)
        print("Not in CSV:", extra_in_dir)
    elif common_files is None:
        print("FILE SUBMISSION FAILED")
        print("No common files found, filenames do not match between os files and metadata list")
    else:
        print("FILE SUBMISSION SUCCEEDED NAME DIR/META NAME TEST")
        #print(f"Shared files:", common_files)

def test_colnames_of_fcs_shared_across_batches():
    col_names_dict = {}
    for file in directory_files:
        if file.endswith('.fcs'):
            file_path = os.path.join(directory_path, file)
            meta, data = fcsparser.parse(file_path, reformat_meta=True)
            fcs_dataframe = pd.DataFrame(data)
            col_names_dict[file] = fcs_dataframe.columns.tolist()
    column_names_df = pd.DataFrame.from_dict(col_names_dict, orient='index')
    for entry in column_names_df
    # then some code to find the row entries per column to see if the strings in in each entry dont match,
        