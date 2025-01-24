import time
import subprocess
import sys
import glob
import time
import os
import random
import pandas as pd
import re
start = time.time()

def data_input(file_type):
    file_path = input(str(f'{file_type} file path:'))
    while not os.path.exists(file_path):
        raise FileNotFoundError("Path not found. Please enter a valid file path.")
    try:
        print('File path accepted')
        file_path = file_path.replace("\\", "/")
        return file_path
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None

meta_data_csv_name = 'fcs_metadata_22012025.csv'# input(str('meta_data_csv_name - with .csv ending:'))
meta_data_location = data_input('meta_data_location:')
meta_data_file = re.sub(r'\.csv$', '', meta_data_csv_name)

panel_file_csv_name = 'panel_metadata_22012025.csv' #input(str('panel_file_csv_name - with .csv ending:'))
panel_file_location = data_input('panel_file_location:')
panel_file_file = re.sub(r'\.csv$', '', panel_file_csv_name)

flow_dir_location = data_input('flow_dir')

# find Rscript.exe
def find_rscript():
    possible_directories = [
        r"C:\Users\mfbx2rdb\AppData\Local\Programs\R"
        r"C:\Program Files\R",                   
        r"C:\Program Files (x86)\R",
        os.environ.get("ProgramFiles"),  
        os.environ.get("ProgramFiles(x86)")
    ]
    for directory in possible_directories:
        if directory:
            rscript_paths = glob.glob(os.path.join(directory, "**", "Rscript.exe"), recursive=True)
            if rscript_paths:
                return rscript_paths[0]
    return None

# Hard exit if there is no Rscript
rscript_path = find_rscript()
if rscript_path is not None:
    print(f"Rscript.exe found at: {rscript_path}")
else:
    print("Rscript.exe not found. Please install R from https://cran.r-project.org/bin/windows/base/ or add Rscript to your PATH.")
    time.sleep(10)
    print('Program will exit, please resume following the install of R.')
    time.sleep(5)
    sys.exit(1)
    
# Seed generation
seed_number = random.seed(1000)

# Data preparation R script
data_prep = f'''
.libPaths("C:/R/win-library/4.3.1")
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Ensure R looks up BioConductor packages
setRepositories(ind = c(1:6, 8))

# List of required packages
required_packages <- c("tidyverse", "remotes")

# Function to check and install missing packages
install_if_missing <- function(packages) {{
  for (pkg in packages) {{
    if (!requireNamespace(pkg, quietly = TRUE)) {{
      install.packages(pkg, lib = "C:/R/win-library/4.3.1")
    }}
  }}
}}

# Check and install missing packages
install_if_missing(required_packages)

# Remove lock directory if it exists
if (dir.exists("C:/R/win-library/4.3.1/00LOCK-htmltools")) {{
  unlink("C:/R/win-library/4.3.1/00LOCK-htmltools", recursive = TRUE)
}}

# Install specific version of htmltools
remotes::install_version(
    "htmltools",
    version = "0.5.7",
    repos = "https://cloud.r-project.org",
    upgrade = "never",
    force = TRUE
)
# Install cyCombine package from GitHub
devtools::install_github("biosurf/cyCombine")

library(cyCombine)
library(tidyverse)
print('Everything loaded in')

# Directory with FCS files
data_dir <- "{flow_dir_location}"

# Extract markers from panel
panel_file <- file.path({panel_file_location}, "{panel_file_csv_name}") # Can also be .xlsx
metadata_file <- file.path({meta_data_location}, "{meta_data_csv_name}") # Can also be .xlsx
print('Meta data loaded')

# Extract markers of interest
markers <- read.csv(panel_file) %>% 
  filter(Type != "none") %>% 
  pull(Antigen)

# Prepare a tibble from directory of FCS files
uncorrected <- prepare_data(
  data_dir = data_dir,
  metadata = metadata_file, 
  filename_col = "Filename",
  batch_ids = "batch",
  condition = "condition",
  down_sample = FALSE,
  markers = markers
)
print('Uncorrected Tibble made')

# Store result in dir
saveRDS(uncorrected, file = file.path(data_dir, "{meta_data_file}_uncorrected.RDS"))
print('Uncorrected RDS Made')
'''
# End of data_prep - write to .R
# Function to write to .R and run using subprocess
def run_rscript(script_to_run, script_name, r_path):
    with open(script_name, "w") as file:
        file.write(script_to_run)
    load_script = [r_path, script_name]
    
    with open(f'{script_name}output.log', 'w') as out, open(f'{script_name}_error.log', 'w') as err:
        subprocess.run(load_script, stdout=out, stderr=err, text=True)

# Run data_prep
run_rscript(data_prep, "data_prep.R", rscript_path)

# Batch correction R script
data_batch_correction = f'''
.libPaths("C:/R/win-library/4.3.1")
options(repos = c(CRAN = "https://cloud.r-project.org/"))
# Ensure R looks up BioConductor packages
setRepositories(ind = c(1:6, 8))
# List of required packages
required_packages <- c("tidyverse", "remotes")

# Function to check and install missing packages
install_if_missing <- function(packages) {{
  for (pkg in packages) {{
    if (!requireNamespace(pkg, quietly = TRUE)) {{
      install.packages(pkg, lib = "C:/R/win-library/4.3.1")
    }}
  }}
}}
# Check and install missing packages
install_if_missing(required_packages)
# Install specific version of htmltools
remotes::install_version(
    "htmltools",
    version = "0.5.7",
    repos = "https://cloud.r-project.org",
    upgrade = "never",
    force = TRUE
)
# Install cyCombine package from GitHub
devtools::install_github("biosurf/cyCombine")


library(cyCombine)
library(tidyverse)

# Load the data
data_dir <- "{flow_dir_location}"
uncorrected <- readRDS(file.path(data_dir, "{meta_data_file}_uncorrected.RDS"))
markers <- get_markers(uncorrected)

# Batch correct
corrected <- batch_correct(
  df = uncorrected,
  covar = "condition",
  markers = markers,
  norm_method = "rank", # "rank" is recommended when combining data with heavy batch effects - otherwise use "scale"
  rlen = 10, # Higher values are recommended if 10 does not appear to perform well
  seed = {seed_number} # Recommended to use your own random seed
)

# Save result
saveRDS(corrected, file.path(data_dir, "{meta_data_file}_corrected.RDS"))
'''

# Run data_batch_correction
run_rscript(data_batch_correction, "data_batch_correction.R", rscript_path)

# Performance evaluation
emd_reduction = ''
mad_score = ''


data_correction_performance = f'''
.libPaths("C:/R/win-library/4.3.1")
options(repos = c(CRAN = "https://cloud.r-project.org/"))
# Ensure R looks up BioConductor packages
setRepositories(ind = c(1:6, 8))
# List of required packages
required_packages <- c("tidyverse", "remotes")

# Function to check and install missing packages
install_if_missing <- function(packages) {{
  for (pkg in packages) {{
    if (!requireNamespace(pkg, quietly = TRUE)) {{
      install.packages(pkg, lib = "C:/R/win-library/4.3.1")
    }}
  }}
}}
# Check and install missing packages
install_if_missing(required_packages)
# Install specific version of htmltools
remotes::install_version(
    "htmltools",
    version = "0.5.7",
    repos = "https://cloud.r-project.org",
    upgrade = "never",
    force = TRUE
)
# Install cyCombine package from GitHub
devtools::install_github("biosurf/cyCombine")


library(cyCombine)
library(tidyverse)

# Load data (if not already loaded)
data_dir <- "{flow_dir_location}"
uncorrected <- readRDS(file.path(data_dir, "{meta_data_file}_uncorrected.RDS"))
corrected <- readRDS(file.path(data_dir, "{meta_data_file}_corrected.RDS"))
markers <- get_markers(uncorrected)

# Re-run clustering on corrected data
labels <- corrected %>% 
  create_som(markers = markers,
             rlen = 10)
uncorrected$label <- corrected$label <- labels

# Evaluate EMD
emd <- evaluate_emd(uncorrected, corrected, cell_col = "label")

# Reduction
{emd_reduction} <- emd$reduction

# Violin plot
emd_violin <- emd$violin
ggsave("data_dir/{meta_data_file}_emd_violin.png", plot = emd_violin, width = 8, height = 6, dpi = 300)


# Scatter plot
emd_scatter <- emd$scatter
ggsave("data_dir/{meta_data_file}_emd_violin.png", plot = emd_violin, width = 8, height = 6, dpi = 300)

# Evaluate MAD
mad <- evaluate_mad(uncorrected, corrected, cell_col = "label")

# Score
{mad_score} <- mad$score
'''

end = time.time()

# Performance Analytics
runtime = round((end - start), 2)
if runtime < 60:
    print(f'Runtime: {runtime} seconds')
else:
    print('Runtime: ' + str(round((runtime/60), 2)) + ' minutes')
print(emd_reduction)
print(mad_score)

performance_dict = {
    "meta_data_file": meta_data_file,
    "runtime": runtime,
    "emd_reduction": emd_reduction,
    "mad_score": mad_score
}

#Dataframe generation of performance analytics
performance_dict = pd.DataFrame([performance_dict])
csv_path = os.path.join(flow_dir_location, f"{meta_data_file}_performance.csv")
performance_dict.to_csv(csv_path, index=False)
