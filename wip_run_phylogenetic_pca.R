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

# Run phylogenetic PCA
phylpca <- phyl.pca(tree, df_data_only_pos, method = "BM", mode = "corr")
phylpca

# Testing
pca_pos <- prcomp(df_data_only_pos)
# Normal biplot
biplot(pca_pos)
# See guide at http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/118-principal-component-analysis-in-r-prcomp-vs-princomp/
# Scree
fviz_eig(pca_pos)
# PCA checking individuals
fviz_pca_ind(pca_pos,
                col.ind = "cos2", # Color by the quality of representation
                gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                repel = TRUE     # Avoid text overlapping
            )
# PCA checking variance contribution
fviz_pca_var(pca_pos,
                col.var = "contrib", # Color by contributions to the PC
                gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                repel = TRUE     # Avoid text overlapping
            )
# Fancy biplot
fviz_pca_biplot(pca_pos, repel = TRUE,
                    col.var = "#2E9FDF", # Variables color
                    col.ind = "#696969"  # Individuals color
                )
# Fancy biplot with groups to show split between
groups <- as.factor(df_data$MDR_bin)
fviz_pca_ind(pca_pos,
                col.ind = groups, # color by groups
                palette = c("#00AFBB",  "#FC4E07"),
                addEllipses = TRUE, # Concentration ellipses
                ellipse.type = "confidence",
                legend.title = "Groups",
                repel = TRUE
            )


# Make use of biplot && screeplot && prcomp() under base R to deal with visualizing PCA (see above with Standard PCA)

# Capture the outputs (eigenvalues & eigenvectors) as dataframes from the phyl.pca output list
# Coerce output dataframes as matrices (data.matrix)
# Calculate the sum along the diagonal of the eigenvalue matrix [sum(diag(matrix))]
# Divide each principle component by the sum of the diagonal to arrive at the percentage of variance explained by that principal component
# Recast the original data along the principal component axes (prob ~ top 10 PC as the variance explained by each PC is low PC1~32%)
# graph (taxa) along the new PC axes. I will need to color for # of variants (circle size) and then a color scale for where the snp occurs (left to right) or by MDR (binary)