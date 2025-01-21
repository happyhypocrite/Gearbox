import subprocess
import sys
from pathlib import Path
import pandas as pd
import glob
import time
import os

meta_data_list = []
panel_file_list = []
file_location_list = []

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

rscript_path = find_rscript()

if rscript_path is not None:
    print(f"Rscript.exe found at: {rscript_path}")
else:
    print("Rscript.exe not found. Please install R from https://cran.r-project.org/bin/windows/base/ or add Rscript to your PATH.")
    time.sleep(10)
    print('Program will exit, please resume following the install of R.')
    time.sleep(5)
    sys.exit(1)
    
############# PART 1 Initial Data Prep #############
# R script as a Python f-string

dataPrep = f'''
options(repos = c(CRAN = "https://cloud.r-project.org/")) 
install.packages("cyCombine")
install.packages("tidyverse")
library(cyCombine)
library(tidyverse)

# Directory with FCS files
data_dir <- "~/data"

# Extract markers from panel
panel_file <- file.path(data_dir, "panel.csv") # Can also be .xlsx
metadata_file <- file.path(data_dir, "metadata.csv") # Can also be .xlsx

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
# Store result
saveRDS(uncorrected, file = file.path(data_dir, "uncorrected.RDS"))