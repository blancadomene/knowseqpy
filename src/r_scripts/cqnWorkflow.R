library(cqn)
library(arrow)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
countsMatrixPath <- args[1]
xPath <- args[2]
annotationPath <- args[3]
outputPath <- args[4]


countsMatrixPath <- 'C:\\Users\\blnc_\\Documents\\TFM-Thesis\\knowseqpy\\r_scripts\\counts.feather'
xPath <- 'C:\\Users\\blnc_\\Documents\\TFM-Thesis\\knowseqpy\\r_scripts\\x.feather'
annotationPath <- 'C:\\Users\\blnc_\\Documents\\TFM-Thesis\\knowseqpy\\r_scripts\\length.feather'
outputPath <- 'C:\\Users\\blnc_\\Documents\\TFM-Thesis\\knowseqpy\\r_scripts\\gene_results.feather'

countsMatrix <- read_feather(countsMatrixPath)
countsMatrix <- tibble::column_to_rownames(countsMatrix, var = "__index_level_0__")
countsMatrix <- as.matrix(countsMatrix)

myGCannot <- read_feather(xPath)
myGCannot <- tibble::column_to_rownames(myGCannot, var = "__index_level_0__")
myGCannot_numeric <- as.numeric(myGCannot$percentage_gene_gc_content)
names(myGCannot_numeric) <- rownames(myGCannot)

mylength <- read_feather(annotationPath)
mylength <- tibble::column_to_rownames(mylength, var = "__index_level_0__")
mylength_numeric <- as.numeric(mylength$Gene_length)
names(mylength_numeric) <- rownames(mylength)

mycqn <- cqn(countsMatrix,
             lengths = mylength_numeric,
             x = myGCannot_numeric,
             sizeFactors = apply(countsMatrix, 2, sum),
             verbose = TRUE)

cqnValues <- mycqn$y + mycqn$offset
expressionMatrix <- cqnValues - min(cqnValues) + 1

expression_df <- as.data.frame(expressionMatrix)
expression_df$row_name <- rownames(expression_df)
write_feather(as.data.frame(expression_df), outputPath)