import os
import pandas as pd
import fcsparser

csv_file_path = input(str('Enter path to .csv of metadata: '))
directory_path = input(str('Enter path to directory containing .fcs files: '))
directory_files = os.listdir(directory_path)


def test_file_size(directory_files, directory_path):
    '''Tests file size of .fcs files in dir - if less that 1kb test fails (indicates .fcs corruption)'''
    file_size_dict = {}
    for file in directory_files:
        if file.endswith('.fcs'):
            fcs_file_path = os.path.join(directory_path, file)
            fcs_file_size = os.path.getsize(fcs_file_path)
            file_size_dict[fcs_file_path] = fcs_file_size
    fcs_file_errored = [key for key, size in file_size_dict.items() if size <= 1]
    if fcs_file_errored:
        print("FILE SUBMISSION FAILED SIZE TEST")
        print("The following file(s) do not meet size criteria: ", fcs_file_errored)
    else:
        print("FILE SUBMISSION SUCCEEDED SIZE TEST")


def test_file_vs_meta(csv_file_path, directory_files):
    '''Reads in the names of the .fcs file names and compare to meta_data file entries to ensure matching entries'''
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
    csv_filenames = df['Filename'].tolist()
    
    common_files = [file for file in csv_filenames if file in directory_files]
    missing_in_dir = [file for file in csv_filenames if file not in directory_files]
    extra_in_dir = [file for file in directory_files if file not in csv_filenames]

    if missing_in_dir or extra_in_dir is not None:
        print("FILE SUBMISSION FAILED META VS DIRECTORY NAME TEST")
        print("Missing in directory:", missing_in_dir)
        print("Not in CSV:", extra_in_dir)
    elif common_files is None:
        print("FILE SUBMISSION FAILED META VS DIRECTORY NAME TEST")
        print("No common files found, filenames do not match between os files and metadata list")
    else:
        print("FILE SUBMISSION SUCCEEDED META VS DIRECTORY NAME TEST")
        #print(f"Shared files:", common_files)


def fcs_colnames_in_dir_to_df(directory_files, directory_path):
    '''Gets column names of all fcs files in fcs file dir - holds data in dataframe'''
    marker_names_dict = {}
    for file in directory_files:
        if file.endswith('.fcs'):
            file_path = os.path.join(directory_path, file)
            meta, data = fcsparser.parse(file_path, reformat_meta=True)
            fcs_file_dataframe = pd.DataFrame(data)
            marker_names_dict[file] = fcs_file_dataframe.columns.tolist()
    column_names_df = pd.DataFrame.from_dict(marker_names_dict, orient='index') #df containing column names of each fcs file with the file name as the key and thus index
    return column_names_df


def test_shared_fcs_colnames_entry(column_names_df):
    '''Finds most_common_marker_name per column in marker_names_dict - finds the specific entries where entry !=most_common_marker_name and reports to user'''
    most_common_marker_name = {} # most_common_marker_name dict: key = column index, value = most common colname (via mode), each dict entry = iterated across each column
    bad_marker_entries = {} # bad_marker_entries dict: key = .fcs file name, value = the bad_entry - can use the most_common_marker_name to understand what it should be.

    for col_index in range(len(column_names_df.columns)):
        most_common = column_names_df.iloc[:, col_index].mode()[0] # most common of a column
        most_common_marker_name[col_index] = most_common #save the most common makername to dict with the index as key, common marker name as value

        mismatches = column_names_df[column_names_df.iloc[:, col_index] != most_common]
        if not mismatches.empty:
            bad_marker_entries[col_index] = mismatches.index.tolist()
            
    '''Report to user what most common marker name is per column, and what fcs files don't meet this standard'''
    if bad_marker_entries:
        #dict comprehension to replace row names
        bad_marker_name_dict = {most_common_marker_name[key]: value for key, value in bad_marker_entries.items()}

        print("FILE SUBMISSION FAILED MARKER NAMING TEST")
        print("Please ensure all flourophore markers are *exactly* the same")
        print("Accepted naming scheme:", most_common_marker_name)
        print('Please find .csv of incorrect markers in the working directory')

        export = pd.DataFrame.from_dict(bad_marker_name_dict, orient='index') #df containing column names of each fcs file with the file name as the key and thus index
        export.to_csv('bad_marker_entries.csv')
    else:
        print("FILE SUBMISSION SUCCEEDED MARKER NAME TEST")
        

# Run code
test_file_size(directory_files, directory_path)
test_file_vs_meta(csv_file_path, directory_files)
column_names_df = fcs_colnames_in_dir_to_df(directory_files, directory_path)
test_shared_fcs_colnames_entry(column_names_df)
