#!/usr/bin/env python3

# Data handling
import numpy as np
import pandas as pd
import sqlite3
import os

"""
By David Brown (db) 20221024

Input:
    Takes an dictionary of filepaths for dataframes containing "Gene" and "Annotation" columns created by finding high feature importances during a Random Forest classification machine learning process.

Execution:
    Identifies genes with missing annotations, e.g. "hypothetical protein". (Often assigned by Roary)
    Queries a previously created SQL database (see https://github.com/microgenomics/tutorials/blob/master/pangenome.md) for the sequences of those genes.

Returns two categories of files.
    1) one multiFASTA file for each gene with a missing annotation. Each multiFASTA file will contain ALL identified sequences assigned to that gene name.
    2) a multiFASTA file containing a single example of each gene, to be passed to a BLAST script.

Files will be organized into subdirectories based on their associated input dataframe filepath.
BLAST sequences will be held under the main directory indicated by the user.

"""
# Filepath to the ranked best predictors per classifier
dict_fp_inputs = {
    "aminoglycosides"               : "/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/graphics_test/hpc_pi_fp_ws_shell_10_distance_1_best_predictors_aminoglycosides_contents_table.tsv",
    "folate_pathway_antagonists"    : "/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/graphics_test/hpc_pi_fp_ws_shell_10_distance_1_best_predictors_folate_pathway_antagonists_contents_table.tsv",
    "macrolides"                    : "/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/graphics_test/hpc_pi_fp_ws_shell_10_distance_1_best_predictors_macrolides_contents_table.tsv",
    "other"                         : "/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/graphics_test/hpc_pi_fp_ws_shell_10_distance_1_best_predictors_other_contents_table.tsv",
    "tetracyclines"                 : "/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/graphics_test/hpc_pi_fp_ws_shell_10_distance_1_best_predictors_tetracyclines_contents_table.tsv",
}
# Filepath to database
fp_to_db = "/scratch/dbrow208/galick_gun_working_dir/subset_900/results_Roary_with_split/database_sub_900_ws.sqlite"

# Indicate a filepath to an output directory
fp_to_out_dir = "/scratch/dbrow208/galick_gun_working_dir/subset_900/results_Roary_with_split/unannotated_gene_fastas"
# Make the directory
os.mkdir(fp_to_out_dir)

# Iterate over each
for classifier_name in dict_fp_inputs.keys():
    # Create a subdirectory within the specified outdirectory
    new_dir_name = classifier_name + "_unannotated_best_predictors"
    this_path = os.path.join(fp_to_out_dir, new_dir_name)
    os.mkdir(this_path)

    # Read in the dataframe
    df_input = pd.read_csv(dict_fp_inputs[classifier_name], sep='\t', low_memory=False)

    # Make database connection and cursor object
    connection = sqlite3.connect(fp_to_db)
    cursor = connection.cursor()

    # Copy working dataframe for safety and identify all 'Gene' names that have a 'hypothetical protein' annotation.
    df_working = df_input.copy()
    df_hypo_prot = df_working[ df_working['Annotation'].str.contains('hypothetical protein') == True ]
    hypo_prot_names = df_hypo_prot['Gene'].to_numpy().tolist()
    # Instantiate an empty list that will the lines of a multiFASTA file
    blast_inputs = []

    # Query database and output query results to files.
    print(hypo_prot_names)
    for name in hypo_prot_names:
        # Query. This format prevents SQL-injection attacks. https://docs.python.org/3/library/sqlite3.html#how-to-use-placeholders-to-bind-values-in-sql-queries
        cursor.execute("""
        select '>'|| cod || '|' || locus_sequence.locus || '|' || pangenoma.gene || x'0a' || sequence
        from locus_sequence
        inner join pangenoma_locus on locus_sequence.locus = pangenoma_locus.locus
        inner join pangenoma on pangenoma_locus.gene = pangenoma.gene
        inner join genomas_locus on locus_sequence.locus = genomas_locus.locus
        where pangenoma.gene = ?
        """, (name,))
        # Execute query. Returns list of tuples.
        db_hits = cursor.fetchall()
        print(db_hits[:3])
        # Formatting for output
        results = [ list(i) for i in db_hits ]
        results = [ i[0] for i in results ]  # Drops the empty final index of the inner lists (former tuples)
        results = [ i.split('\n') for i in results ]
        results = [ item for sublist in results for item in sublist ]   # Unpack sublists
        #results = [ i.rstrip('\n') for i in results ]   # Strip extra newline characters
        results = [ i + "\n" for i in results ]     # Make certain each string in the list has a newline
        print(results[:6])
        # Save results to the previously determined output directory.
        out_file_name = fp_to_out_dir + "/" + new_dir_name + "/" + name + "_sequences.fasta"
        with open(out_file_name, 'w') as outfile:
            outfile.writelines(results)
        # Capture a single result for BLAST in a tuple (<FASTA HEADER>, <FASTA SEQUENCE>)
        blast_inputs.extend( results[:2] )

    connection.close()

    out_blast_name = fp_to_out_dir + "/" + classifier_name + "_sequences_for_blast.fasta"
    with open(out_blast_name, 'w') as outblast:
        outblast.writelines(blast_inputs)

###NOTE also do the RAxML tree matrix calc && try that polish thing
# donig raxml tree, doing databasebuilder
# try polish thing next