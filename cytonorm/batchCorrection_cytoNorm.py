import typing
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias as _TypeAlias
    typing.TypeAlias = _TypeAlias
    
import flowsom
import cytonormpy as cnp
import os
import pandas as pd

# File wrangling and input
input_directory = input(str("Path to .fcs files: "))
import glob
fcs_files = glob.glob(os.path.join(input_directory, "*.fcs"))
if not fcs_files:
    print("No .fcs files found in the input directory!")
else:
    print(f"Found {len(fcs_files)} .fcs files.")
    
output_directory = os.path.join(input_directory, "cytonorm_output")
if not os.path.exists(output_directory):
    os.mkdir(output_directory) 
metadata_location = input(str("Path to metadata .csv: "))
metadata = pd.read_csv(metadata_location)

print(f"Metadata needs columns: 'file_name', 'reference', 'batch', 'sample_ID'\n {metadata.head} \n This is what you have")

cn_object = cnp.CytoNorm() # generate the cytonorm object
transformer = cnp.AsinhTransformer(cofactors = 5)
flowsom = cnp.FlowSOM(n_clusters = 4)
cn_object.add_transformer(transformer)
cn_object.add_clusterer(flowsom)

coding_detectors = pd.read_csv(input_directory + "\\coding_detectors.txt", header = None)[0].tolist()
cn_object.run_fcs_data_setup(input_directory = input_directory,
                      metadata = metadata,
                      channels = coding_detectors,
                      output_directory = output_directory,
                      prefix = "Norm",
                      truncate_max_range = False)
cn_object.run_clustering(cluster_cv_threshold = 2)
cn_object.calculate_quantiles()
cn_object.calculate_splines(goal = "batch_mean")
cn_object.normalize_data(n_jobs = 1)
