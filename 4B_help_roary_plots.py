#!/usr/bin/env python3

"""
Step 4B. Run after Roary, but before roary_plots.py to generate some plots.

TODO:
"""

import numpy as np
import pandas as pd

fp_reduced = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/results_RAxML_no_split/core_gene_alignment.aln.reduced"
fp_gene_table = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/results_Roary_no_split/gene_presence_absence.csv"
fp_out = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/results_Roary_no_split/gene_presence_absence.csv.reduced"

acc_pres = []
with open(fp_reduced, "r") as f1:
    these_lines = f1.readlines()
    these_lines = these_lines[1:]
    acc_pres = [ row.split()[0] for row in these_lines ]


df_gene_table = pd.read_csv(fp_gene_table, sep=",", low_memory=False)
these_cols = list(df_gene_table.columns)
print(these_cols)
acc_total = [ i for i in these_cols if "GCF" in i ]

acc_pres = set(acc_pres)
acc_total = set(acc_total)

acc_missing = acc_total - acc_pres
acc_missing = list(acc_missing)

df_out = df_gene_table.copy()
df_out.drop( labels=acc_missing, axis="columns", inplace=True )
df_out.to_csv(fp_out)
"""
open raxml
open gene_presence_absence.csv

read csv to pandas
???transpose csv

# best solution
parse raxml for dupes (grep for IMPORTANT WARNING: Sequences GCF_000005845 and GCF_000269645 are exactly identical)

# temp solution
parse the "*.reduced" alignment file in phylip, line by line and if not present, then remove from below

drop columns or post transpose rows from csv
save and run roary plots
"""