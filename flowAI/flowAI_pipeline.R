if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("flowAI")

require(flowAI)

setwd('C:\\Users\\mfbx2rdb\\OneDrive - The University of Manchester\\PDRA\\Sequencing\\Py scripts\\Projects\\ImmAcc\\gearbox_data\\Batch_fcsdump') #Wkdir of where the fcs files are
fcsfiles <- dir(".", pattern="*fcs$") #regex to pick out the .fcs files specifically.

GbLimit <- 8    # decide the limit in gigabyte for your batches of FCS files
size_fcs <- file.size(fcsfiles)/1024/1024/1024    # it calculates the size in gigabytes for each FCS file
groups <- ceiling(sum(size_fcs)/GbLimit)
cums <- cumsum(size_fcs)
batches <- cut(cums, groups) 


for(i in 1:groups){
  flow_auto_qc(fcsfiles[which(batches == levels(batches)[i])], output = 0) 
}

