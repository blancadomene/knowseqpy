# DifferentialExpressionLimma.R
# Purpose: Performs differential expression analysis using limma package (lmFit and eBayes)
# Inputs: expression_file (CSV), design_file (CSV), output_table_file (CSV)
# Output: Writes topTable results to a CSV file

library(arrow)
library(limma)

args <- commandArgs(trailingOnly = TRUE)
expressionMatrixPath <- args[1]
designMatrixPath <- args[2]
outputTablePath <- args[3]
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

fit <- lmFit(expressionMatrix, designMatrix)
fit <- eBayes(fit)

topTable <- topTable(fit, number = maxGenes, coef = 2, sort.by = "logFC", p.value = pvalue, adjust = "fdr", lfc = lfc)
DEGsMatrix <- expressionMatrix[rownames(topTable),]
DEGsMatrix <- DEGsMatrix[unique(rownames(DEGsMatrix)),]
# cvDEGsResults[[i]] <- list(topTable,DEGsMatrix)
# cvDEGsList[[i]] <- rownames(table)
# result <- list(DEGsTable = list(table,DEGsMatrix), DEGsMatrix = rownames(table))
# result <- data.frame(DEGsMatrix)
DEGsMatrix$row_name <- rownames(DEGsMatrix)
write_feather(DEGsMatrix, outputTablePath)

aaaaa <- read_feather(outputTablePath)