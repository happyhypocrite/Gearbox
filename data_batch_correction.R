
.libPaths("./R/win-library/4.3.1")
options(repos = c(CRAN = "https://cloud.r-project.org/")) 
# To ensure Rstudio looks up BioConductor packages run:
setRepositories(ind = c(1:6, 8))
install.packages("tidyverse")
if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes")
}
remotes::install_version(
    "htmltools",
    version = "0.5.7",
    repos = "https://cloud.r-project.org",
    upgrade = "never",
    force = TRUE
)
# Then install package with
devtools::install_github("biosurf/cyCombine")

library(cyCombine)
library(tidyverse)

# Load the data
data_dir <- "C:/Users/mfbx2rdb/OneDrive - The University of Manchester/PDRA/Sequencing/Py scripts/Projects/ImmAcc/Gearbox/StrokeIMPaCT_SmartTube_V2/StrokeIMPaCT_SmartTube123_v2/"
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
