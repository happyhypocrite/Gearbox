
.options(repos = c(CRAN = "https://cloud.r-project.org/"))

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

# Load data (if not already loaded)
data_dir <- "C:/Users/mfbx2rdb/OneDrive - The University of Manchester/PDRA/Sequencing/Py scripts/Projects/ImmAcc/Gearbox/StrokeIMPaCT_SmartTube_V2/StrokeIMPaCT_SmartTube123_V2/"
uncorrected <- readRDS(file.path(data_dir, "fcs_metadata_22012025_uncorrected.RDS"))
corrected <- readRDS(file.path(data_dir, "fcs_metadata_22012025_corrected.RDS"))
markers <- get_markers(uncorrected)

# Re-run clustering on corrected data
labels <- corrected %>% 
  create_som(markers = markers,
             rlen = 10)
uncorrected$label <- corrected$label <- labels

# Evaluate EMD
emd <- evaluate_emd(uncorrected, corrected, cell_col = "label")

# Reduction
emd_reduction <- emd$reduction

# Violin plot
emd_violin <- emd$violin
ggsave("data_dir/fcs_metadata_22012025_emd_violin.png", plot = emd_violin, width = 8, height = 6, dpi = 300)


# Scatter plot
emd_scatter <- emd$scatter
ggsave("data_dir/fcs_metadata_22012025_emd_violin.png", plot = emd_violin, width = 8, height = 6, dpi = 300)

# Evaluate MAD
mad <- evaluate_mad(uncorrected, corrected, cell_col = "label")

# Score
mad_score <- mad$score

#Create a JSON for the value to save
cat(jsonlite::toJSON(list(mad_score = mad_score, emd_reduction = emd_reduction)))
