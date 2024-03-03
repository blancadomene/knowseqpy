# SVA_BatchCorrectionWorkflow.R
# Purpose: Performs batch effect correction on gene expression data using the sva package.
# This script reads expression data and sample labels, estimates and adjusts for surrogate variables (SVs) to correct for batch effects.
# Input:
#   expression_data.feather (Feather) - Path to the gene expression data file in Feather format.
#   design_matrix.feather (Feather) - Path to the labels file in Feather format, used to construct model matrices.
#   batch_results.feather (Feather) - Path where the batch corrected expression matrix will be saved in Feather format.
# Output: Writes the batch corrected expression matrix to a Feather file.

library(arrow)
library(sva)

# Parse command line arguments
args <- commandArgs(trailingOnly <- TRUE)
expressionMatrixPath <- args[1]
labelsPath <- args[2]
outputPath <- args[3]

expressionMatrixPath <- 'C:\\Users\\blnc_\\Documents\\TFM-Thesis\\knowseqpy\\r_scripts\\expression_data.feather'
labelsPath <- 'C:\\Users\\blnc_\\Documents\\TFM-Thesis\\knowseqpy\\r_scripts\\design_matrix.feather'
outputPath <- 'C:\\Users\\blnc_\\Documents\\TFM-Thesis\\knowseqpy\\r_scripts\\batch_results.feather'


expressionMatrix <- read_feather(expressionMatrixPath)
expressionMatrix <- as.data.frame(tibble::column_to_rownames(expressionMatrix, var = "external_gene_name"))
expressionMatrix <- as.matrix(expressionMatrix)
labels <- read_feather(labelsPath)$`Sample.Type`

if(is.character(labels)){
  labels <- as.factor(labels)
}

# Constructs model matrices
mod <- model.matrix(~labels)
mod0 <- model.matrix(~1, data <- labels)

# Estimates the number of surrogate variables (SVs) to adjust for in the batch correction
n.sv <- num.sv(expressionMatrix,mod,method<-"leek")

# Performs Surrogate Variable Analysis (SVA) to identify SVs and adjust the expression matrix
svobj <- sva(expressionMatrix,mod,mod0,n.sv<-n.sv)
ndb <- dim(expressionMatrix)[2]
nmod <- dim(mod)[2]
n.sv<-svobj$n.sv
mod1 <- cbind(mod,svobj$sv)
gammahat <- (expressionMatrix %*% mod1 %*% solve(t(mod1) %*% mod1))[,(nmod+1):(nmod + svobj$n.sv)]
expressionMatrixBatchCorrected <- expressionMatrix - gammahat %*% t(svobj$sv)

# Write corrected matrix to file
expressionMatrixBatchCorrected_df <- as.data.frame(expressionMatrixBatchCorrected)
expressionMatrixBatchCorrected_df$row_name <- rownames(expressionMatrixBatchCorrected_df)
write_feather(as.data.frame(expressionMatrixBatchCorrected_df), outputPath)