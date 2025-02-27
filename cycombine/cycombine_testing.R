options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Ensure R looks up BioConductor packages
setRepositories(ind = c(1:6, 8))

# List of required packages
required_packages <- c("tidyverse", "remotes", "devtools")

# Function to check and install missing packages
install_if_missing <- function(packages) {{
  for (pkg in packages) {{
    if (!requireNamespace(pkg, quietly = TRUE)) {{
      install.packages(pkg)
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
  upgrade = "always",
  force = TRUE
)
# Install cyCombine package from GitHub
devtools::install_github("biosurf/cyCombine")

library(cyCombine)
library(tidyverse)
print('Everything loaded in')

# Directory with FCS files
data_dir <- "C:/Users/mfbx2rdb/OneDrive - The University of Manchester/PDRA/Sequencing/Py scripts/Projects/ImmAcc/gearbox_data/Batch_fcsdump/"

# Extract markers from panel
panel_file <- paste0("C:/Users/mfbx2rdb/OneDrive - The University of Manchester/PDRA/Sequencing/Py scripts/Projects/ImmAcc/gearbox_data/metadata/","panel_metadata_22012025.csv") # Can also be .xlsx
metadata_file <- paste0("C:/Users/mfbx2rdb/OneDrive - The University of Manchester/PDRA/Sequencing/Py scripts/Projects/ImmAcc/gearbox_data/metadata/", "fcs_metadata_22012025.csv") # Can also be .xlsx
print('Meta data loaded')

# Extract markers of interest
markers <- read.csv(panel_file) %>% 
  filter(Type != "None") %>% 
  pull(Antigen)

#uncorrected <- prepare_data(
#  data_dir = data_dir,
#  metadata = metadata_file, 
#  filename_col = "Filename",
#  batch_ids = "batch",
#  sample_ids = "Patient_id",
#  condition = "condition",
#  down_sample = FALSE,
#  derand = FALSE,
#  markers = markers,
#  transform = FALSE
#)

# Prepare a tibble from directory of FCS files
uncorrected <- prepare_data(
                      data_dir = data_dir,
                      metadata = metadata_file,
                      filename_col = "Filename",
                      batch_ids = "batch",
                      condition = "condition",
                      sample_ids = "Patient_id",
                      markers = markers,
                      down_sample = FALSE,
                      .keep = FALSE,
                      clean_colnames = FALSE,
                      panel = panel_file,
                      panel_channel = "Channel",
                      panel_antigen = "Antigen",
                      transform = TRUE)

print('Uncorrected Tibble made')



# Store result in dir
saveRDS(uncorrected, file = file.path(data_dir, "uncorrected.RDS"))
print('Uncorrected RDS Made')



corrected <- batch_correct(
  df = uncorrected,
  covar = "condition",
  markers = markers,
  norm_method = "scale", # "rank" is recommended when combining data with heavy batch effects
  rlen = 10, # Higher values are recommended if 10 does not appear to perform well
  seed = 101 # Recommended to use your own random seed
)


saveRDS(corrected, file.path(data_dir, "corrected.RDS"))

emd <- evaluate_emd(uncorrected, corrected, cell_col = "label")
