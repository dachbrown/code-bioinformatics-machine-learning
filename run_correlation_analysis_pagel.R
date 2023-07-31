#!/usr/bin/env Rscript

# Test for correlation between binary characters using Pagel's 1994 method
# by David Brown (db) 20221208
# Modified from code by Liam J. Revell found at:
# http://www.phytools.org/Cordoba2017/ex/9/Pagel94-method.html

### WIP
### Current Issues
# None, appears to complete; begin loop testing

library(feather)
library(phytools)

# Define variables for input filepaths
#fp_in_tree <- "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/results_RAxML_no_split/ns_900_core_opt_outgroup_rooted.nw"
fp_in_tree <- "/Users/dbrow208/Documents/dissertation_final/correlation_analysis_R_pagel/ns_900_core_opt_outgroup_rooted.pruned.nw"
fp_in_tsv  <- "/Users/dbrow208/Documents/dissertation_final/correlation_analysis_R_pagel/ns_prot_mutS.original.tsv"

# Define variables for output filepaths
fp_out_ftr <- "/Users/dbrow208/Documents/dissertation_final/correlation_analysis_R_pagel/output_pagel.ftr"
fp_out_tsv <- "/Users/dbrow208/Documents/dissertation_final/correlation_analysis_R_pagel/output_pagel.tsv"

# Read tree
tree <- read.tree(fp_in_tree)
print(tree)
#print(tree$tip.label)
these_labels <- tree$tip.label
test_this <- unique(these_labels)
print(length(test_this))

# Read TSV file
df_data <- read.csv(fp_in_tsv, row.names = 1, header = TRUE, sep = "\t")
print(dim(df_data))
# Reformat the row names to match the tree end nodes
rownames(df_data)   <-  lapply(rownames(df_data), function(x) {
                            gsub(" ", "_", x)})
#rownames(df_data)   <-  paste(rownames(df_data), "_", sep = "")
this_diff <- setdiff(test_this, rownames(df_data))
print(this_diff)
#print(rownames(df_data))

# Test implement Pagel
x <- setNames(df_data$pos_002, rownames(df_data))
y <- setNames(df_data$MDR_bin, rownames(df_data))
fit_xy <- fitPagel(tree, y, x)
print(fit_xy)
