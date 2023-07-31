#!/usr/bin/env Rscript

# Test for phylogenetic PCA
# by David Brown (db) 20230112
# Modified from code by Liam J. Revell found at:
# http://blog.phytools.org/2018/11/computing-phylogenetic-pca-scores-for.html

### WIP
### Current Issues
# See commented additions at the end to begin adding correct transformations to the data when deciding on principal components
# Think about color/viz options for MDR and/or for SNP group patterns

library(dplyr)
library(feather)
library(phytools)
library(factoextra)

# Define variables for input filepaths
fp_in_tree <- "ns_900_core_opt_outgroup_rooted.pruned.nw"
fp_in_tsv  <- "ns_prot_mutS.original.tsv"

# Read tree
tree <- read.tree(fp_in_tree)
test_this <- unique(tree$tip.label)

# Read TSV file
df_data <- read.csv(fp_in_tsv, row.names = 1, header = TRUE, sep = "\t")
print(dim(df_data))
# Reformat the row names to match the tree end nodes
rownames(df_data)   <-  lapply(rownames(df_data), function(x) {
                            gsub(" ", "_", x)})
# Compare row names to confirm
this_diff <- setdiff(test_this, rownames(df_data))
print(this_diff)

# Select only the SNP position columns from the dataframe.
df_data_only_pos <- select(df_data, contains("pos_"))
df_data_snp_pos <- select(df_data, c("snp_var", contains("pos_")))
df_data_meta <- select(df_data, -c("snp_var", contains("pos_")))

