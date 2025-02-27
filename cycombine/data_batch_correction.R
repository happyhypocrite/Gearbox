
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Ensure R looks up BioConductor packages
setRepositories(ind = c(1:6, 8))

# List of required packages
required_packages <- c("tidyverse", "remotes")

# Function to check and install missing packages
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
  }
}

# Check and install missing packages
install_if_missing(required_packages)

library(cyCombine)
library(tidyverse)

# Load the data
data_dir <- "C:/Users/mfbx2rdb/OneDrive - The University of Manchester/PDRA/Sequencing/Py scripts/Projects/ImmAcc/gearbox_data/Batch_fcsdump/"
uncorrected <- readRDS(file.path(data_dir, "fcs_metadata_22012025_uncorrected.RDS"))
markers <- get_markers(uncorrected)

# Batch correct
corrected <- batch_correct(
  df = uncorrected,
  covar = "condition",
  markers = markers,
  norm_method = "rank", # "rank" is recommended when combining data with heavy batch effects - otherwise use "scale"
  rlen = 10, # Higher values are recommended if 10 does not appear to perform well
  seed = None # Recommended to use your own random seed
)

# Save result
saveRDS(corrected, file.path(data_dir, "fcs_metadata_22012025_corrected.RDS"))
