
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
print('Everything loaded in')

# Directory with FCS files
data_dir <- "C:/Users/mfbx2rdb/OneDrive - The University of Manchester/PDRA/Sequencing/Py scripts/Projects/ImmAcc/Gearbox/StrokeIMPaCT_SmartTube_V2/StrokeIMPaCT_SmartTube123_v2/"

# Extract markers from panel
panel_file <- file.path(C:/Users/mfbx2rdb/OneDrive - The University of Manchester/PDRA/Sequencing/Py scripts/Projects/ImmAcc/Gearbox/StrokeIMPaCT_SmartTube_V2/, "panel_metadata_22012025.csv") # Can also be .xlsx
metadata_file <- file.path(C:/Users/mfbx2rdb/OneDrive - The University of Manchester/PDRA/Sequencing/Py scripts/Projects/ImmAcc/Gearbox/StrokeIMPaCT_SmartTube_V2/, "fcs_metadata_22012025.csv") # Can also be .xlsx
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
saveRDS(uncorrected, file = file.path(data_dir, "fcs_metadata_22012025_uncorrected.RDS"))
print('Uncorrected RDS Made')
