# LimmaDEGsExtractionWorkflow.R
# Purpose: Performs differential expression analysis using the limma package.
# This script uses the lmFit and eBayes methods to identify differentially expressed genes.
# Input:
#   expression_file.feather (Feather) - Path to the expression data file in Feather format.
#   design_file.feather (Feather) - Path to the design matrix file in Feather format.
#   output_table_file.feather (CSV) - Path to save the output table in Feather format.
#   pvalue (numeric) - P-value threshold for significance.
#   lfc (numeric) - Log fold change threshold for significance.
#   maxGenes (integer/Inf) - Maximum number of genes to report. Use "inf" for no limit.
# Output: Writes topTable results, filtered by the specified thresholds, to a Feather file.

library(arrow)
library(limma)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
expressionMatrixPath <- args[1]
designMatrixPath <- args[2]
outputPath <- args[3]
pvalue <- as.numeric(args[4])
lfc <- as.numeric(args[5])
if (args[6] == "inf") {
    maxGenes <- Inf
} else {
    maxGenes <- as.integer(args[6])
}

expressionMatrix <- read_feather(expressionMatrixPath)
expressionMatrix <- tibble::column_to_rownames(expressionMatrix, var = "__index_level_0__")
designMatrix <- read_feather(designMatrixPath)
designMatrix <- tibble::column_to_rownames(designMatrix, var = "__index_level_0__")

# Fit linear models and estimate empirical Bayes statistics
fit <- lmFit(expressionMatrix, designMatrix)
fit <- eBayes(fit)

# Identify and extract top differentially expressed genes based on specified criteria
topTable <- topTable(fit, number = maxGenes, coef = 2, sort.by = "logFC", p.value = pvalue, adjust = "fdr", lfc = lfc)
DEGsMatrix <- expressionMatrix[rownames(topTable),]
DEGsMatrix <- DEGsMatrix[unique(rownames(DEGsMatrix)),]

# Write extracted DEGs to file
DEGsMatrix$row_name <- rownames(DEGsMatrix)
write_feather(DEGsMatrix, outputPath)