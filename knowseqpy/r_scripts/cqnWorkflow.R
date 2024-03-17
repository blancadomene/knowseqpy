# CQN_NormalizationWorkflow.R
# Purpose: Performs normalization of gene expression data using the Conditional Quantile Normalization (CQN) method.
# This script reads counts matrix, GC content, and gene length information, applies CQN normalization, and saves the normalized expression matrix.
# Input:
#   countsMatrixPath (Feather) - Path to the counts matrix file in Feather format. Rows represent genes, and columns represent samples.
#   xPath (Feather) - Path to the file containing GC content for each gene in Feather format. Must contain a column for percentage GC content.
#   annotationPath (Feather) - Path to the file containing annotation data, including gene length, in Feather format.
#   outputPath (Feather) - Path where the normalized expression matrix will be saved in Feather format.
# Output: Writes the CQN normalized expression matrix to a Feather file. The output matrix has genes as rows and samples as columns, adjusted for GC content and gene length.

library(cqn)
library(arrow)

# Parse command line arguments for input and output file paths
args <- commandArgs(trailingOnly = TRUE)
countsMatrixPath <- args[1]
xPath <- args[2]
annotationPath <- args[3]
outputPath <- args[4]

# Read the counts matrix and convert to a matrix format
countsMatrix <- read_feather(countsMatrixPath)
countsMatrix <- tibble::column_to_rownames(countsMatrix, var = "__index_level_0__")
countsMatrix <- as.matrix(countsMatrix)

# Read the GC content file, extract the numeric GC content, and assign names
myGCannot <- read_feather(xPath)
myGCannot <- tibble::column_to_rownames(myGCannot, var = "__index_level_0__")
myGCannot_numeric <- as.numeric(myGCannot$percentage_gene_gc_content)
names(myGCannot_numeric) <- rownames(myGCannot)

# Read the gene length file, extract numeric gene lengths, and assign names
mylength <- read_feather(annotationPath)
mylength <- tibble::column_to_rownames(mylength, var = "__index_level_0__")
mylength_numeric <- as.numeric(mylength$Gene_length)
names(mylength_numeric) <- rownames(mylength)

# Apply CQN normalization using counts, gene lengths, and GC content
mycqn <- cqn(countsMatrix,
             lengths = mylength_numeric,
             x = myGCannot_numeric,
             sizeFactors = apply(countsMatrix, 2, sum),
             verbose = TRUE)

# Prepare and write the normalized expression matrix to Feather format
cqnValues <- mycqn$y + mycqn$offset
expressionMatrix <- cqnValues - min(cqnValues) + 1
expression_df <- as.data.frame(expressionMatrix)
expression_df$row_name <- rownames(expression_df)
write_feather(as.data.frame(expression_df), outputPath)
