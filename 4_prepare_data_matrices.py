#!/usr/bin/env python3

import numpy as np
import pandas as pd
import feather

"""
From /Users/dbrow208/Documents/galick_gun/test_prokka_roary/roary_to_ml/convert_roary_outputs_to_ml_features.py

Convert Roary outputs & NCBI data into combined feature/label matrix for machine learning. - db 20220830

Features are from Roary (gene pres/abs and eventually the genomic code)
Labels will be NCBI data on AMR classes, AMR uniques, AMR prefixes, AMR counts

Consider /Users/dbrow208/Documents/galick_gun/ml_sklearn_test/convert_roary_matrix_to_ml.py


TNT
- medium priority - deal with the missing data bug at bottom??
- low priority - consider updating "pangenome" logic to accept multiple levels of selection ie both "core" & "soft_core"
- high priority - connect somehow with the snp-sites output of a multifasta? (should be its own script. there will be weirdness if a gene is not found in all isolates)
- ************ high priority - checkm (LINE 342) - Add columns for 'contamination', 'completeness', 'heterogeneity' Search "JOIN QC HERE"
-- Need the precise name and input filetype of the checkm results
-- Need an input line, filter for high complete (>95) and low contamination (<5) then reset index and join

####
# Bug, for some reason "GCA_001677475" is in the NCBI file, but not the Roary file. Something must have gone wrong on the HPC? Check error logs.
# Bug?? if you choose RefSeq over GenBank, be sure to cast from 'GCA' to 'GCF'

# Potential issues
- in dictionary "drug_classes"
-- "arm" could refer to widely different mechanisms in CARD, depending on species
-- "abc" is inherently multidrug, but the NARMS test would likely show it as cefotaxime resistant => cephem

"""
#####
# FUNCTIONS
# Define a function to expand (from sublists) a given dataframe with presence/absence (binarized) data columns.
def makeBinaryCols(this_df, str_df_key_col, str_df_search_col):
    """
    Input:
    - The dataframe (pandas dataframe)
    - The key column name (string)
    - The column name to be searched (string)
    
    Output:
    - The altered dataframe with new columns representing the presence/absence of unique targets within the specified search column

    """
    df_search_col_as_list = this_df[str_df_search_col].to_list()                    # Cast column series to list
    all_col_vals = [ str(item) for item in df_search_col_as_list if item != None ]  # Remove nulls and cast values to string
    all_col_vals = [ item.split(',') for item in all_col_vals ]                     # Split value strings into sublists (if they exist)
    uniq_col_vals = [ item for sublist in all_col_vals for item in sublist ]        # Unpack sublists. Similar to .expand() instead of .append()
    uniq_col_vals = sorted(list(set(uniq_col_vals)))                                # Use set() to arrive at unique strings, cast to list, and sort the list.
    """# Original. Created error similar to this https://github.com/pandas-dev/pandas/issues/42477
    for target in uniq_col_vals:
        this_df[target] = this_df[str_df_key_col].map(  # Map a dictionary of the new column onto the dataframe given the key column
            dict(                                       # Create the dictionary
                zip(                                    # Call zip on two lists
                    this_df[str_df_key_col].to_list(),  # Dictionary keys are from the key column
                    [ 1 if target in str(cell_data).split(',') else 0 for cell_data in this_df[str_df_search_col].to_list() ]  # List comprehension to binarize the presence/absence of a target feature within the specified column
                    # str(cell_data) is necessary to handle NoneType, and still convert that as absence (0).
                    # str(cell_data).split(',') is necessary to handle exact matches with list logic instead of string.
                    # For example, "tet" is in both "tet" and "tet(A)", but not in ["tet(A)"].
                )
            )
        )
    """
    list_objs_to_concat = [this_df] # Create a list to hold Series objects for pd.concat
    for target in uniq_col_vals:
        temp_dict = dict(                                       # Create the dictionary
                        zip(                                    # Call zip on two lists
                            this_df[str_df_key_col].to_list(),  # Dictionary keys are from the key column
                            [ int(1) if target in str(cell_data).split(',') else int(0) for cell_data in this_df[str_df_search_col].to_list() ]  # List comprehension to binarize the presence/absence of a target feature within the specified column
                            # str(cell_data) is necessary to handle NoneType, and still convert that as absence (0).
                            # str(cell_data).split(',') is necessary to handle exact matches with list logic instead of string.
                            # For example, "tet" is in both "tet" and "tet(A)", but not in ["tet(A)"].
                        )
                    )
        temp_series = pd.Series(temp_dict, name=target)     # Assign the Series and name it
        list_objs_to_concat.append(temp_series)
    #print(list_objs_to_concat)
    this_df = pd.concat( list_objs_to_concat, axis=1 )
    return(this_df)

# Define a function to expand (from sublists) a given dataframe with presence/absence (binarized) data columns for the prefixes.
def makePrefixBinaryCols(this_df, str_df_key_col, str_df_search_col, list_targets, prefix_length, new_col_name):
    """
    Input:
    - The dataframe (pandas dataframe)
    - The key column name (string)
    - The column name to be searched (string)
    - A list containing the targets (list of strings)
    - The number of letters in the prefix (integer)
    - Name of new prefix column (string)
    
    Output:
    - The altered dataframe with new columns representing the presence/absence of targets within the specified search column

    """
    # For every sublist within a cell of the column, change the items of the sublist to only the first characters as defined by prefix_length.
    prefix_only = []
    for cell in this_df[str_df_search_col].to_list():   # Cast column to list
        cell = str(cell)                                # Confirm the list items are all strings (to cover NoneTypes)
        cell = cell.split(',')                          # Within each cell, split the string into a sublist
        cell = [ 'None' if i == 'None' else i[:prefix_length] for i in cell ]      # Capture the prefix of each item in the sublist
        cell = ",".join(cell)                           # Rejoin the prefixes to a string
        prefix_only.append(cell)                        # Append to placeholder list
    
    # Add the prefix column to the dataframe
    prefix_col = dict(zip(this_df[str_df_key_col].to_list(), prefix_only))
    prefix_col = pd.Series(prefix_col, name=new_col_name)
    #print(prefix_only)
    """
    for target in list_targets:
        new_target_name = str(target) + "_family"
        this_df[new_target_name] = this_df[str_df_key_col].map(  # Map a dictionary of the new column onto the dataframe given the key column
            dict(                                       # Create the dictionary
                zip(                                    # Call zip on two lists
                    this_df[str_df_key_col].to_list(),  # Dictionary keys are from the key column
                    [ 1 if target in str(cell_data).split(',') else 0 for cell_data in prefix_only ]  # List comprehension to binarize the presence/absence of a target feature within the specified column
                    # str(cell_data) is necessary to handle NoneType, and still convert that as absence (0).
                    # str(cell_data).split(',') is necessary to handle exact matches with list logic instead of string.
                    # For example, "tet" is in both "tet" and "tet(A)", but not in ["tet(A)"].
                )
            )
        )
    """
    list_objs_to_concat = [ this_df, prefix_col ] # Create a list to hold Series objects for pd.concat
    for target in list_targets:
        new_target_name = str(target) + "_family"
        temp_dict = dict(                                       # Create the dictionary
                        zip(                                    # Call zip on two lists
                            this_df[str_df_key_col].to_list(),  # Dictionary keys are from the key column
                            [ 1 if target in str(cell_data).split(',') else 0 for cell_data in prefix_only ]  # List comprehension to binarize the presence/absence of a target feature within the specified column
                            # str(cell_data) is necessary to handle NoneType, and still convert that as absence (0).
                            # str(cell_data).split(',') is necessary to handle exact matches with list logic instead of string.
                            # For example, "tet" is in both "tet" and "tet(A)", but not in ["tet(A)"].
                        )
                    )
        temp_series = pd.Series(temp_dict, name=new_target_name)     # Assign the Series and name it
        list_objs_to_concat.append(temp_series)
    #print(list_objs_to_concat)
    this_df = pd.concat( list_objs_to_concat, axis=1 )
    return(this_df)

# Define a function to check for existing/missing data in a column and return presence/absence (1/0)
def detectPresAbs(df_col):
    df_col = df_col.fillna(0)
    new_vals = [ 0 if val == 0 else 1 for val in df_col.to_list() ]
    return(new_vals)

# Define a function to take values from a column and assign alphabetical characters for phylogenetic inference.
def makePhyloCharacters(this_df, str_df_key_col, str_df_search_col, new_col_name):
    """
    20220927 - db
    DEPRECATED - This logic was to represent phylogenetic characters with more than two states, i.e. "letter" and 26 states. Was to be applied to "sets" of MDR resistance.
        Two reasons for deprecation,
            - with currently 10 types of drug resistance tested by NARMS, and 1 type of "other" resistance added by me, there are 165 combinations of MDR sets with 11 choose 3. Does not include, 11 choose 4, 11 choose 5, etc. 
            - this mathematical progression would prove especially unwieldy if NARMS changes or if more resistance granularity is desired
    
    Best practice appears to be remaining at a SNP matrix or binary matrix level. Each character should have binary states.


    Input:
    - The dataframe (pandas dataframe)
    - The key column name (string)
    - The column name to be searched (string)
    - The new column name (string)
    
    Output:
    - The altered dataframe with new column of the phylogenetic characters
    - A reference dataframe with keys/values for traits/characters

    """
    import string

    df_search_col_as_list = this_df[str_df_search_col].to_list()                        # Cast column series to list
    all_col_groups = [ str(item) for item in df_search_col_as_list if item != None ]    # Remove nulls and cast values to string
    uniq_col_sets = sorted(list(set(all_col_groups)))                                   # Use set() to arrive at unique, unduplicated strings, cast to list, and sort the list.
    alphabet = list(string.ascii_uppercase)
    alphabet = [ alphabet[i] for i in range(0, len(uniq_col_sets)) ]
    reference_dict = dict(zip(uniq_col_sets, alphabet))                     # Make dictionary reference
    reference_df = pd.DataFrame.from_dict(reference_dict, orient='index')   # Make dataframe reference
    this_df[new_col_name] = this_df[str_df_key_col].map(    # Map a dictionary of the new column onto the dataframe given the key column
        dict(                                               # Create the dictionary
            zip(                                            # Call zip on two lists
                this_df[str_df_key_col].to_list(),          # Dictionary keys are from the key column
                [ reference_dict[cell_data] for cell_data in this_df[str_df_search_col].to_list() ] # Capture the alphabetical character instead of the trait
            )
        )
    )
    return(this_df, reference_df)

#####
# INPUTS - USER
# Designate input filepaths
#input_ncbi  = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/hpc_test_20220829/metadata_subset_asm_level_Scaffold_geo_loc_USA_CA_standardized_1012.ftr"
input_ncbi = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/metadata_subset_species_taxid_562_num_samples_911.ftr"
#input_roary = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/hpc_test_20220829/results/gene_presence_absence.csv"
# Choose one of the below and ADJUST 'output_name' ACCORDINGLY
input_roary = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/results_Roary_no_split/gene_presence_absence.csv"   # No split paralogs
#input_roary = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/results_Roary_with_split/gene_presence_absence.csv" # With split paralogs
input_pass_qc = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/qc_table_pass_complete_99.0_contam_1.0.ftr"
# Designate the pangenome cutoff from roary
pangenome_type = "shell_10"    # Choose from one of the "pangenome_dict" keys ('core', 'soft_core', 'shell', 'cloud', 'total', 'core_and_soft_core', 'soft_core_and_shell', 'shell_and_cloud', 'exclude_cloud', 'core_10', 'shell_10')
# Designate output filepath
output_fp = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/final_confirmation/"
output_name = "test_fc_subset_900_no_split"

#####
# PREPARE NCBI DATA
# Read in the subset (created by Step 1) as a pandas dataframe
df_ncbi = pd.read_feather(input_ncbi)

# Create a working copy of the NCBI dataframe
df_labels = df_ncbi.copy()
#print(df_labels.head())
print("No. of unique accessions with version :", len(pd.unique(df_labels['asm_acc'])))
# Reformat the ASM accession by dropping the version suffix and changing the 'GCA' to 'GCF'
df_labels['asm_acc'] = df_labels['asm_acc'].apply(lambda x : x.split('.')[0].replace('A', 'F'))
# ALSO BE SURE TO CHANGE FROM GCA TO GCF IF YOU CHOSE REFSEQ
print("No. of unique accessions drop version :", len(pd.unique(df_labels['asm_acc'])))
print("The above 2 lines should be equivalent. They check for loss of duplicates when dropping the ASM accession version suffix.")

# Within the copy of the NCBI dataframe, drop all columns unneccessary for prediction as labels.
df_labels.drop(
    labels=[                    # Uncomment to drop, comment to keep. PLEASE CHECK YOUR COMMAS WITHIN THE LIST.
        'index',                # A previous index.
        '#label',               # An identifier from NCBI Pathogen Detection.
        'HHS_region',
        'LibraryLayout',
        'PFGE_PrimaryEnzyme_pattern',
        'PFGE_SecondaryEnzyme_pattern',
        'Platform',             # Sequencing platform
        'Run',                  # Sequencing run number
        #'asm_acc',              # Assembly accession, necessary key for joining and identifying data.
        #'asm_level',            # Assembly level, may be kept for filtering or dropped.
        'asm_stats_contig_n50',
        'asm_stats_length_bp',  # Assembly stats.
        'asm_stats_n_contig',
        'assembly_method',
        'attribute_package',
        'bioproject_acc',
        'bioproject_center',
        'biosample_acc',
        'collected_by',         # Collector information.
        'collection_date',      # Collection date, removed due to standardization below.
        #'epi_type',             # Sample origin data, a necessary label.
        'fullasm_id',           # Full assembly identifier for NCBI, removed due to use of 'asm_acc' as key.
        'geo_loc_name',         # Geographic location, removed due to standardization below.
        'host',                 # Host, removed due to standardization below.
        'host_disease',         # Host disease if recorded, may be kept for filtering or dropped. NEEDS STANDARDIZATION.
        'isolation_source',     # Specific isolation source ('feces'), may be kept for filtering or dropped. NEEDS STANDARDIZATION.
        'lat_lon',
        'outbreak',             # Outbreak if recorded.
        'sample_name',          # Unstandardized sample name.
        #'scientific_name',      # Scientific name, may be kept for filtering or dropped.
        'serovar',              # Serovar if recorded.
        'species_taxid',        # NCBI standardized taxonomic identification for species.
        'sra_center',
        'sra_release_date',
        'strain',               # Strain if recorded.
        'target_acc',
        'target_creation_date',
        'taxid',                # NCBI standardized taxonomic identifier.
        'wgs_acc_prefix',       # Whole genome sequence accession prefix identifying project as uploaded to NCBI.
        'wgs_master_acc',       # Whole genome sequence master accession for an upload with some assembly.
        'minsame',
        'mindiff',
        'number_drugs_resistant',
        'number_drugs_intermediate',
        'number_drugs_susceptible',
        'number_drugs_tested',
        #'number_amr_genes',     # Number of AMR genes, a necessary label.
        'AST_phenotypes'       # Antimicrobial susceptibility testing information.
        #'AMR_genotypes',        # Identified AMR genotypes, a necessary label.
        #'collection_datetime',  # Datetime transformation of 'collection_date'.
        #'host_v2',              # Host data after cleaning/standardizing, a necessary label.
        #'geo_loc_name_v2'       # Geographic location data after cleaning/standardizing, a necessary label.
    ],
    axis="columns",
    inplace=True
)

# Set the index without dropping the column, as it is used in future.
df_labels.set_index('asm_acc', inplace=True, drop=False)

#####
# BINARIZE GENOTYPE FEATURES
# Use the unique AMR_genotypes and expand the dataframe to a binary matrix with each unique as its own column
df_labels = makeBinaryCols(df_labels, "asm_acc", "AMR_genotypes")
#print(df_labels.head())
#print(df_labels.columns.to_list())
#print("No. of unique values :", len(pd.unique(df_labels['asm_acc'])))
#print(df_labels.tet.to_numpy)

#####
# BINARIZE PREFIX FEATURES
# Within the copy of the NCBI dataframe, identify all unique values for the "AMR_genotypes" column
col_amr_genotypes = df_labels.AMR_genotypes.to_list()
all_amr_genotypes = [ str(item) for item in col_amr_genotypes if item != None ]     # Remove nulls and cast to string
all_amr_genotypes = [ item.split(',') for item in all_amr_genotypes ]               # Split AMR genotypes strings into sublists
uniq_amr_genotypes = [ item for sublist in all_amr_genotypes for item in sublist ]  # Unpack sublists. Similar to .expand() instead of .append()
uniq_amr_genotypes = sorted(list(set(uniq_amr_genotypes)))                          # Use set() to arrive at unique strings, cast to list, and sort the list.
# Within the copy of the NCBI dataframe, identify all unique 3-letter prefixes from the "AMR_genotypes" column
uniq_amr_prefix = [ i[:3] for i in uniq_amr_genotypes ] # Subset the strings to the 3 letter family prefix.
uniq_amr_prefix = sorted(list(set(uniq_amr_prefix)))    # Use set() to arrive at unique strings, cast to list, and sort the list.

# Use the unique 3-letter prefix AMR_genotypes and expand the dataframe to a binary matrix with each prefix as its own column
df_labels = makePrefixBinaryCols(df_labels, "asm_acc", "AMR_genotypes", uniq_amr_prefix, 3, "grouped_AMR_prefixes")
#print(df_labels.head)
df_labels = df_labels.copy()
#####
# BINARIZE MDR FEATURES AS DRUG CLASSES (>=3)
# Using the "AMR_genotypes" column, determine if there are 3 or more resisted classes of drug
# Dictionary with gene families as key and the CLSI drug class that gene family key resists as value. Alphabetical by values.
# Much of this was done by hand, could be automated in future. Searched the prefix in CARD, then selected key/value from "Drug Class" field.
## Please cross reference:
### CLSI classes and examples https://www.cdc.gov/narms/antibiotics-tested.html
### CARD to connect resistance genes and CLSI classes https://card.mcmaster.ca/home
#### Potential further references at https://card.mcmaster.ca/ontology/36008
drug_classes = {
    "aac"   :   "aminoglycosides",
    "aad"   :   "aminoglycosides",
    "ant"   :   "aminoglycosides",
    "aph"   :   "aminoglycosides",
    "arm"   :   "aminoglycosides",  # From CARD https://card.mcmaster.ca/ontology/37238 NOTE: armA is in E. coli, while armR is in Pseudomonas. COULD BE A PROBLEM.
    "rmt"   :   "aminoglycosides",
    "bla"   :   "beta_lactam_combination_agents",
    #   :   "cephems",              # From CARD https://card.mcmaster.ca/ontology/36415 "Cephems have a six-membered dihydrothiazine ring with a sulfur atom and double bond fused with its beta-lactam ring. This group includes the cephalosporins and cephamycins, the latter containing an additional alpha-methoxy group."
    "abc"   :   "cephems",          # A judgment call, as family abc are multidrug efflux pumps. https://card.mcmaster.ca/ontology/40695 NARMS tests with cephem resistance with cefotaxime, so abc would "hit" on that screen even though it also resists other drugs.
    "dfr"   :   "folate_pathway_antagonists",
    "sul"   :   "folate_pathway_antagonists",
    "ere"   :   "macrolides",
    "erm"   :   "macrolides",
    "mef"   :   "macrolides",
    "mph"   :   "macrolides",
    "msr"   :   "macrolides",
    #   :   "monobactams",          # From CARD https://card.mcmaster.ca/ontology/35923 "Monobactams are a class of beta-lactam antibiotics with a broad spectrum of antibacterial activity, and have a structure which renders them highly resistant to beta-lactamases. Unlike penams and cephems, monobactams do not have any ring fused to its four-member lactam structure."
    "sat"   :   "nucleosides",      # Uncertain if CARD "nucleoside antibiotic" https://card.mcmaster.ca/ontology/36174 is CLSI recognized
    #   :   "penems",               # From CARD https://card.mcmaster.ca/ontology/40360 "Penems are a class of unsaturated beta-lactam antibiotics with a broad spectrum of antibacterial activity and have a structure which renders them highly resistant to beta-lactamases. All penems are all synthetically made and act by inhibiting the synthesis of the peptidoglycan layer of bacterial cell walls."
    "amp"   :   "penicillins",
    "cat"   :   "phenicols",
    "flo"   :   "phenicols",
    #   :   "quinolones",           # From CARD https://card.mcmaster.ca/ontology/35920 See "Synonym(s)"
    "qep"   :   "quinolones",
    "qnr"   :   "quinolones",
    "tet"   :   "tetracyclines",
    "arr"   :   "other",            # Uncertain if CARD "rifampin ADP-ribosyltransferase (Arr)" https://card.mcmaster.ca/ontology/36529 is CLSI recognized or NARMS tested
    "ble"   :   "other",            # Uncertain if CARD "glycopeptide antibiotic" https://card.mcmaster.ca/ontology/36220 is CLSI recognized or NARMS tested
    "cml"   :   "other",            # Uncertain if CARD "major facilitator superfamily (MFS) antibiotic efflux pump" https://card.mcmaster.ca/ontology/36003 [see Sub-Term(s)] is CLSI recognized or NARMS tested
    "fos"   :   "other",            # Uncertain if CARD "fosfomycin thiol transferase" https://card.mcmaster.ca/ontology/36272 is CLSI recognized or NARMS tested
    "lnu"   :   "other",            # Uncertain if CARD "lincosamide nucleotidyltransferase (LNU)" https://card.mcmaster.ca/ontology/36360 is CLSI recognizd or NARMS tested
    "mcr"   :   "other",            # Uncertain if CARD "peptide antibiotic" https://card.mcmaster.ca/ontology/36192 [see Sub-Term(s)] is CLSI recognized or NARMS tested
    "oqx"   :   "other",            # Uncertain if CARD "resistance-nodulation-cell division (RND) antibiotic efflux pump" https://card.mcmaster.ca/ontology/36005 is CLSI recognized or NARMS tested
    "qac"   :   "other",            # Uncertain if CARD "disinfecting agents and antiseptics" https://card.mcmaster.ca/ontology/43746 is CLSI recognized or NARMS tested
    "None"  :   "no_resistance"     # Explicitly catches samples without any recorded resistance
}

# Transform the AMR prefixes (families) to classes of antimicrobial resistance compounds
resist_drug_classes = []
for resist_prefixes in df_labels['grouped_AMR_prefixes'].to_list():
    resist_prefixes = resist_prefixes.split(',')
    resist_prefixes = [ drug_classes[key] for key in resist_prefixes ]
    resist_prefixes = sorted(list(set(resist_prefixes)))
    resist_drug_classes.append(resist_prefixes)

# Make a column of the drug resistance classes for each isolate
df_labels['resistance_classes'] = [ ",".join(sublist) for sublist in resist_drug_classes ] # Note that 'resist_drug_classes' is a list.
# Represent MDR as binary presence/absence of greater than 3 classes of drug resistance
df_labels['MDR_bin'] = [ 1 if len(sublist) >= 3 else 0 for sublist in resist_drug_classes ] # Note that 'resist_drug_classes' is a list.
# Make a column of the MDR classes, each MDR class is a string of 3 or more resistance classes
#df_labels['MDR_classes'] = [ ",".join(sublist) if len(sublist) >= 3 else "no_MDR" for sublist in resist_drug_classes ] # Note that 'resist_drug_classes' is a list.

### TEST NOTE: attempt to reduce the number of MDR classes by subsetting. The fact that aminoglycoside && beta_lactam are frequent and alphabetical is coincidental but fortuitous. - Fixed 20221015 db, see below.
# Now returns a string of all classes of resistance if there are 3 or more. Each string could be useful for multiclass classification in ML if encoded or binarized.
df_labels['MDR_classes'] = [ ",".join(sublist) if len(sublist) >= 3 else "no_MDR" for sublist in resist_drug_classes ] # Note that 'resist_drug_classes' is a list of lists.
### A data subset for presence/absence of three specific drug class resistances.
df_labels['amiblafol_bin'] = [ 1 if ("aminoglycosides" in sublist) and ("beta_lactam_combination_agents" in sublist) and ("folate_pathway_antagonists" in sublist) else 0 for sublist in resist_drug_classes ]
### An attempt to treat "bla" or "beta_lactam_combination_agents" as part of the ground truth, and therefore remove it on the assumption that it is wild-type. In the 900(817) subset, it was only absent ~40 times.
mdr_classes_drop_bla = [ [item for item in sublist if item != 'beta_lactam_combination_agents'] for sublist in resist_drug_classes ]    # Nested list comprehensions
df_labels['MDR_classes_drop_bla'] = [ ",".join(sublist) if len(sublist) >= 3 else "only_bla_or_no_MDR" for sublist in mdr_classes_drop_bla ]    # Similar logic to 'MDR_classes', but preserves the list typing for now.
# Add a column to the dataframe based on the prevalence of the 'MDR_classes_drop_bla' to allow removal for ML
#dict_mdr_class_prevalence =  df_labels['MDR_classes_drop_bla'].value_counts().to_dict()
#dict_mdr_to_keep = { key:('keep' if value >=8 else 'drop') for key, value in dict_mdr_class_prevalence.items() }
#df_labels['ML_multiclass_filter'] = [ dict_mdr_to_keep[cell] for cell in df_labels['MDR_classes_drop_bla'].to_numpy().tolist() ]

#####
# BINARIZE MDR CLASSES AS INDIVIDUAL FEATURES
# Use the unique "resistance_classes" and expand the dataframe to a binary matrix with each unique resistance_class as its own column
df_labels = makeBinaryCols(df_labels, "asm_acc", "resistance_classes")
#print(df_labels)

#### DEPRECATED - 20220927 db
# BINARIZE MDR SETS AS CHARACTERS
# Assigne phylogenetic characters to each grouping (or set) of observed MDR drug class
#df_labels, df_phylo_character = makePhyloCharacters(df_labels, "asm_acc", "MDR_classes", "MDR_phylo_characters")
#df_phylo_character.to_csv(output_fp + pangenome_type + "_phylo_characters.tsv", sep="\t")

#####
# PREPARE QC DATA
# Read in the QC output file "" as a pandas dataframe
df_pass_qc = pd.read_feather(input_pass_qc)
df_qc = df_pass_qc.copy()
df_qc_new_index = df_qc['Bin Id'].to_list()
# Format the full identifiers to the correct prefix
df_qc_new_index = [ i.split('.')[0] for i in df_qc_new_index ]
df_qc['asm_acc'] = df_qc_new_index
df_qc.drop(
    labels=[            # Please reference https://github.com/Ecogenomics/CheckM/wiki/Genome-Quality-Commands#qa
        "index",            # The column name that occurs if an index is reset. Common for feather tables, as they require a numeric index.
        "Bin Id",           # The full identifying name
        "Marker lineage",   # The marker lineage
        "# genomes",        # The number of genomes
        "# markers",        # The number of markers
        "# marker sets",    # The number of marker sets
        "0",                # The number of times each marker gene is identified
        "1",                # The number of times each marker gene is identified
        "2",                # The number of times each marker gene is identified
        "3",                # The number of times each marker gene is identified
        "4",                # The number of times each marker gene is identified
        "5+"	            # The number of times each marker gene is identified
    ],
    axis="columns",
    inplace=True
)
df_qc.set_index('asm_acc', inplace=True)
total_isolates_pass_qc = df_qc.shape[0]
print('\nThe use of the quality file ' + input_pass_qc + '\nresulted in a reduction from ' + str(df_labels.shape[0]) + ' to ' + str(total_isolates_pass_qc) + ' accessions that passed quality control.')


#####
# PREPARE ROARY DATA
# Read in the Roary output file "gene_presence_absence.csv" as a pandas dataframe
df_roary = pd.read_csv(input_roary, sep=",", low_memory=False)

# Create a copy of the Roary dataframe and specify index
df_features = df_roary.copy()
# SUBSET BASED OFF CORE, CLOUD HERE remove all 100 and then keep 15 & above b/c of soft core & shell
# NOTE: Inclusive values core == 99-100%; soft core == 95-98%; shell == 15-94%; cloud == 0-14%
#df_features = df_features[ df_features['No. isolates'].between(14, 100, inclusive="neither") ] # Original setting
# Key is pangenome segment, while value is [lower, upper, inclusive]
pangenome_dict = {
    'core'                  :   [0.99, 1.00, "both"],
    'soft_core'             :   [0.95, 0.99, "left"],
    'shell'                 :   [0.15, 0.95, "left"],
    'cloud'                 :   [0.00, 0.15, "left"],
    'total'                 :   [0.00, 1.00, "both"],
    'core_and_soft_core'    :   [0.95, 1.00, "both"],
    'soft_core_and_shell'   :   [0.15, 0.99, "left"],
    'shell_and_cloud'       :   [0.00, 0.95, "left"],
    'exclude_cloud'         :   [0.15, 1.00, "both"],
    'core_10'               :   [0.10, 1.00, "both"],
    'shell_10'              :   [0.10, 0.95, "left"]

}
df_features = df_features[ df_features['No. isolates'].between(pangenome_dict[pangenome_type][0] * total_isolates_pass_qc, pangenome_dict[pangenome_type][1] * total_isolates_pass_qc, inclusive=pangenome_dict[pangenome_type][2]) ]

#df_features = df_features[ df_features['No. isolates'].between(123, 776, inclusive=pangenome_dict[pangenome_type][2]) ]

# Within the copy of the Roary dataframe, drop all columns between "Gene" and the first identifying accession column
df_features.drop(
    # The first 14 columns given by Roary as output. The remaining columns are isolate (accession) identifiers.
    labels=[                                # Uncomment to drop, comment to keep. PLEASE CHECK YOUR COMMAS WITHIN THE LIST.
        #"Gene",                             # Gene name. Will be transposed from rows to columns, necessary to keep.
        "Non-unique Gene name",             # Other names if "Gene" is not unique
        "Annotation",                       # Description of gene function
        "No. isolates",                     # Number of isolates (accessions) where a gene was identified
        "No. sequences",                    # Number of unique sequences where a gene was identified
        "Avg sequences per isolate",        # Mean number of identified sequences per isolate
        "Genome Fragment",                  # Location within a fragmented genome? (NOT LOCUS)
        "Order within Fragment",            # Location within the above fragment?
        "Accessory Fragment",               # Identity of the fragment?
        "Accessory Order with Fragment",    # ???
        "QC",                               # Quality control data. From kraken?
        "Min group size nuc",               # Minimum number of nucleotides identified as the gene
        "Max group size nuc",               # Maximum number of nucleotides identified as the gene
        "Avg group size nuc"                # Mean number of nucleotides identified as the gene
    ],
    axis="columns",
    inplace=True
)

# Within the copy of the Roary dataframe, change the values within the accession columns from the "accession_locus" naming convention to binary pres/abs
df_features['Gene'] = df_features['Gene'].astype('string')  # Prepare for transpose by casting 
df_features[df_features.columns[1:]] = df_features[df_features.columns[1:]].apply(detectPresAbs, axis=0)

#print(df_features.head)
#print(df_features.dtypes)
print( "\nNo. of unique genes :", len(pd.unique(df_features['Gene'])) )
print( "Compare the above with the '" + pangenome_type + "' selection in the Roary file 'summary_statistics.txt' associated with " + input_roary)

# Set the index, transpose the copied, altered Roary dataframe
df_features.set_index('Gene', inplace=True)
df_features_T = df_features.transpose()

#####
# JOIN DATA
# Make the "asm_acc" column the index for the NCBI dataframes (confirm number of rows)
df_labels.set_index('asm_acc', inplace=True)

# Join the NCBI and quality control (QC) dataframes on the index (accessions) or on a column with index as key
df_labels_qc = df_labels.join(df_qc, how='left')
#print(df_labels_qc)
# Join the NCBI/QC and Roary dataframes on the index (accessions) or on a column with index as key
df_out = df_features_T.join(df_labels_qc, how='left', lsuffix='_feature')    # If a duplicate gene name is present, specify the left hand one as the feature (as opposed to "_family"), just in case.
#print(df_out.grouped_AMR_prefixes)
#print(df_out.columns.to_list())
print("The final dataframe has dimensions " + str(df_out.shape))

#####
# CREATE DATA OUTPUTS
# Save the output dataframe as a combined feature/label matrix with columns as observed data and rows as accessions samples
df_out = df_out.reset_index()   # Feather can throw errors if the index is not numeric
output_fp_ftr = output_fp + output_name + "_" +  pangenome_type + ".ftr"
output_fp_tsv = output_fp + output_name + "_" + pangenome_type + ".tsv"
df_out.to_feather(output_fp_ftr)
df_out.to_csv(output_fp_tsv, sep='\t')

print(df_out['MDR_classes_drop_bla'].value_counts())
print(df_out['MDR_classes_drop_bla'].describe())
#print(df_out['ML_multiclass_filter'].value_counts())
#print(df_out['ML_multiclass_filter'].describe())

#practice_df = df_out[ df_out['ML_multiclass_filter'] == 'keep' ]
#print(practice_df['MDR_classes_drop_bla'].value_counts())
#print(practice_df['MDR_classes_drop_bla'].describe())

"""
# Previous check statements

print(df_out.tet.describe())
print(df_out.tet.value_counts())
print(df_out['tet(A)'].describe())
print(df_out['tet(A)'].value_counts())
print(df_out['tet(B)'].describe())
print(df_out['tet(B)'].value_counts())
print(df_out['tet_family'].describe())
print(df_out['tet_family'].value_counts())
print('not_MDR' in df_out.columns.to_list())
print(df_out.columns.to_list())
print(df_out.MDR_phylo_characters)
print(phylo_character_dict)
print(df_out.MDR_bin)
print(df_out.AMR_prefix.to_list())
print(df_out.MDR_bin.to_list())
print(df_out.MDR_classes.to_list())
print(df_out.AMR_genotypes.to_list())
"""