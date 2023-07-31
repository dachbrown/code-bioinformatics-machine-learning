#!/usr/bin/env python3

# Data handling
import numpy as np
import pandas as pd
import sqlite3
import os

"""
By David Brown (db) 20230418 - Modification of process_search_hypotheticals.py to conform with committee request to reannotate Prokka/Roary assigned annotations
- Added a "specified_genes" list of notable genes to investigate with BLAST (~45)
- Added logic to handle the "specified_genes" list
- Removed logic to eliminate the within +/- 5% length filter for sanity (might need to update manuscript wording)

Input:
    Takes an dictionary of filepaths for dataframes containing "Gene" and "Annotation" columns created by finding high feature importances during a Random Forest classification machine learning process.

Execution:
    Identifies genes with missing annotations, e.g. "hypothetical protein". (Often assigned by Roary) # Note this file does the opposite, identifying a present annotation so the sequence can be confirmed via BLAST - DB 20230417
    Queries a previously created SQL database (see https://github.com/microgenomics/tutorials/blob/master/pangenome.md) for the sequences of those genes.

Returns two categories of files.
    1) one multiFASTA file for each gene with a missing annotation. Each multiFASTA file will contain ALL identified sequences assigned to that gene name.
    2) a multiFASTA file containing a single example of each gene, to be passed to a BLAST script.

Files will be organized into subdirectories based on their associated input dataframe filepath.
BLAST sequences will be held under the main directory indicated by the user.


"""
# Filepath to the ranked best predictors per classifier
dict_fp_inputs = {
    "aminoglycosides"               : "/scratch/dbrow208/galick_gun_working_dir/clean_run_20221130/TRAINING_num_clust_66_ranked_predictors_aminoglycosides_contents_table.tsv",
    "folate_pathway_antagonists"    : "/scratch/dbrow208/galick_gun_working_dir/clean_run_20221130/TRAINING_num_clust_66_ranked_predictors_folate_pathway_antagonists_contents_table.tsv",
    "macrolides"                    : "/scratch/dbrow208/galick_gun_working_dir/clean_run_20221130/TRAINING_num_clust_66_ranked_predictors_macrolides_contents_table.tsv",
    "tetracyclines"                 : "/scratch/dbrow208/galick_gun_working_dir/clean_run_20221130/TRAINING_num_clust_66_ranked_predictors_tetracyclines_contents_table.tsv",
    "other"                         : "/scratch/dbrow208/galick_gun_working_dir/clean_run_20221130/TRAINING_num_clust_66_ranked_predictors_other_contents_table.tsv",
}
# Filepath to database
fp_to_db = "/scratch/dbrow208/galick_gun_working_dir/subset_900/results_Roary_with_split/database_sub_900_ws.sqlite"

# Indicate a filepath to an output directory
fp_to_out_dir = "/scratch/dbrow208/galick_gun_working_dir/20230418_hypo_prots_sanity"
# Make the directory
os.mkdir(fp_to_out_dir)

# Indicate threshold values for to include the top ranks (top rank is 0-based for python, but give the actual number, eg 3 returns ranks 0,1,2). Used 6 for the VAL set, 5 for the TRAINING set. Could vary based on the permutation importance values. - db 20221118
rank_dict = {
    "aminoglycosides"               : 5,
    "folate_pathway_antagonists"    : 5,
    "macrolides"                    : 5,
    "tetracyclines"                 : 5,
    "other"                         : 5,
}

# List of "Genes" with Roary assigned names, supplied here to check the Roary annotations for comparison - DB 20230417
specified_genes = [
    "umuC_2",
    "traM",
    "traY",
    "group_2526",
    "ccdB",
    "traA",
    "yfjQ_4",
    "group_5903",
    "yfjJ",
    "group_4579",
    "cbeA_1",
    "yeeS_1",
    "php_1",
    "group_684",
    "group_257",
    "group_1508",
    "dbpA",
    "sieB",
    "ydaF",
    "rzpR",
    "group_11293",
    "group_25593",
    "ydaV",
    "trkG",
    "tfaR",
    "pinR",
    "ydaM",
    "ydaG",
    "intR",
    "ydbH",
    "racC",
    "ydaE",
    "ynaE",
    "recE",
    "ydaW",
    "ydbK",
    "ompN",
    "ydaT",
    "group_10760",
    "kilR",
    "recT",
    "group_10767",
    "group_37342",
    "group_18678",
    "rcbA"
]

# Instantiate a placeholder for 
blast_candidates = {}
# Iterate over each
for classifier_name in dict_fp_inputs.keys():
    # Create a subdirectory within the specified outdirectory
    new_dir_name = classifier_name + "_top_" + str(rank_dict[classifier_name]) + "_unannotated_ranked_predictors"
    this_path = os.path.join(fp_to_out_dir, new_dir_name)
    os.mkdir(this_path)

    # Read in the dataframe
    df_input = pd.read_csv(dict_fp_inputs[classifier_name], sep='\t', low_memory=False)

    # Copy working dataframe for safety and identify all 'Gene' names that have a 'hypothetical protein' annotation.
    df_working = df_input.copy()
    # Identify all "hypothetical proteins" of a given rank
    #df_hypo_prot = df_working[ (df_working['Annotation'].str.contains('hypothetical protein') == True) & (df_working['rank'] < rank_dict[classifier_name]) ]
    # Added line to filter for only specific genes - DB 20230417
    df_hypo_prot = df_working[ df_working['Gene'].isin(specified_genes) ]
    # Identify all "hypothetical proteins" of a given rank # Modified True -> False for reannotation - DB 20230417
    #df_hypo_prot = df_hypo_prot[ (df_hypo_prot['Annotation'].str.contains('hypothetical protein') == False) & (df_hypo_prot['rank'] < rank_dict[classifier_name]) ] # Remove if only checking specified genes - DB 20230418
    
    hypo_prot_names = df_hypo_prot['Gene'].to_numpy().tolist()

    # Instantiate an empty list that will the lines of a multiFASTA file
    #blast_inputs = []

    # Query database and output query results to files.
    print(hypo_prot_names)

    
    for name in hypo_prot_names:
        # Make database connection and cursor object
        connection = sqlite3.connect(fp_to_db)
        cursor = connection.cursor()
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
        #print(db_hits[:3])
        connection.close()
        # Formatting for output
        results = [ list(i) for i in db_hits ]
        results = [ i[0] for i in results ]  # Drops the empty final index of the inner lists (former tuples)
        results = [ i.split('\n') for i in results ]
        results = [ item for sublist in results for item in sublist ]   # Unpack sublists
        #results = [ i.rstrip('\n') for i in results ]   # Strip extra newline characters
        results = [ i + "\n" for i in results ]     # Make certain each string in the list has a newline
        #print(results[:6])
        # Save results to the previously determined output directory.
        out_file_name = fp_to_out_dir + "/" + new_dir_name + "/" + name + "_sequences.fasta"
        with open(out_file_name, 'w') as outfile:
            outfile.writelines(results)
        # Capture a single result for BLAST in a tuple (<FASTA HEADER>, <FASTA SEQUENCE>)
        #blast_inputs.extend( results[:2] )

        """
        # Set a flag
        candidate_found = False
        # Identify the average group size nucleotides for the given hypothetical
        col_val = df_hypo_prot.loc[df_hypo_prot['Gene'] == name, 'Avg group size nuc'].item()
        # Find the first header/sequence pair where the sequence (including a "\n") is within +/- 5% of the average length
        for pos in range(0, len(results), 2):
            if len(results[pos+1]) > 0.95*col_val and len(results[pos+1]) < 1.05*col_val and candidate_found == False and name not in blast_candidates.keys():
                blast_candidates[name] = results[pos:pos+2]
                candidate_found = True
        """
        # Added to ignore the length filter for a sanity check - DB 20230418
        blast_candidates[name] = results[:2]


    #out_blast_name = fp_to_out_dir + "/" + classifier_name + "_sequences_for_blast.fasta"
    #with open(out_blast_name, 'w') as outblast:
    #    outblast.writelines(blast_inputs)
blast_outputs = []
for key in blast_candidates.keys():
    blast_outputs.extend(blast_candidates[key])
segments = int(np.ceil(len(blast_outputs) / 40))
for val in range(0,segments):
    #out_blast_name = fp_to_out_dir + "/all_sequences_for_blast_segment_0" + str(val) + ".fasta"
    out_blast_name = fp_to_out_dir + "/confirm_annotate_sequences_for_blast_segment_0" + str(val) + ".fasta"    # Added to show difference - DB 20230417
    with open(out_blast_name, 'w') as outblast:
        outblast.writelines(blast_outputs[val*40:(val+1)*40])

###NOTE also do the RAxML tree matrix calc && try that polish thing
# donig raxml tree, doing databasebuilder
# try polish thing next