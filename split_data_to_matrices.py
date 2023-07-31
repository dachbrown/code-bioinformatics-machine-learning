#!/usr/bin/env python3

# Data handling
import numpy as np
import pandas as pd

"""
Split input data frame into only the feature subset.
"""

####################
### FUNCTIONS
####################
# Splits a dataframe into featues and labels, based on a reference column.
def splitFeaturesLabels(filepath_to_data, reference_col, index_estimate):
    """
    Splits a feather format dataframe/table of both features and labels into two separate pandas dataframes, based on a reference column. Also returns original dataframe.
    
    INPUTS
    "filepath_to_data"  A string filepath for the input.
    "reference_col"     A string column name, represents the (not inclusive) last column of the features dataframe and the (inclusive) first column of the labels dataframe.
    "index_estimate"    A negative integer estimate of the position of the "reference_col" in the input data. Speeds up dataframe indexing, based on the data set and ratio of features to labels. Suggested value -400.

    OUTPUTS
    "these_features"    A dataframe of the features.
    "these_labels"      A dataframe of the labels, with the first column as the "reference_col".
    "this_data"         The original, unsplit dataframe for sanity checks.
    """
    this_data = pd.read_feather(filepath_to_data)
    #data.set_index('index', inplace=True)   # Changes index from the numeric required by feather, back to the original "asm_id" now called "index" due to the '.reset_index()' call.

    # Designate the first column of the labels.
    mark_index = this_data.columns.to_list().index(reference_col, index_estimate)

    # Identify column names for the features and subset
    these_feature_names = this_data.columns.to_list()[:mark_index]
    these_features = this_data[these_feature_names]
    #print(features.head)
    these_label_names = this_data.columns.to_list()[mark_index:]
    these_labels = this_data[these_label_names]

    # Remove the index column from the features. This is an artifact of pandas ".reset_index()" when saving to feather format, which REQUIRES a numeric row index instead of string indices.
    these_features = these_features.drop(columns=['index'])
    return(these_features, these_labels, this_data)

# Print the ".describe()" and ".value_counts()" from pandas.
def showColStats(dataframe, col_name):
    """
    Shows useful column stats. Returns nothing.
    """
    print("\n", col_name)
    print(dataframe[col_name].describe())
    print(dataframe[col_name].value_counts())
####################
####################


####################
### USER INPUTS - for specific parameters
####################

# Choose the designator for the input filepath (see below dictionary "dict_fp")
designator = 'fp_ws_shell_10'
# Set the filepath to the directory containing the feather format inputs
this_dir_fp = '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts'
# Assign a filepath to the output directory
fp_to_out = "/Users/dbrow208/Documents/galick_gun/clean_run_20221025/attempt_3"
# Choose a column to investigate basic metrics
this_col = "MDR_classes_drop_bla" # Also "MDR_classes", "MDR_bin"
# Choose columns for a machine learning multilabel analysis (Non-exclusive targets, where each target will be fitted by a single estimator. Full model is a combination of estimators.)
these_multilabels = [
    'aminoglycosides',
    #'beta_lactam_combination_agents',  # Dropped as overrepresented (nearly 100% prevalence)
    #'cephems',                         # Dropped as underrepresented (nearly 0% prevalence)
    'folate_pathway_antagonists',
    'macrolides',
    #'nucleosides',                     # Dropped as underrepresented (nearly 0% prevalence)
    #'penicillins',                     # Dropped as underrepresented (nearly 0% prevalence)
    #'quinolones',                      # Dropped as when removing the lower prevalence columns, there were no longer any occurrences of 'quinolones' positive values (1)
    'tetracyclines',
    'other',
    #'no_resistance'                    # Dropped as underrepresented (nearly 0% prevalence)
]


####################
### FILEPATHS TO DATASETS
####################
# All filepaths
dict_fp = {
    "fp_ws_ec"      : this_dir_fp + '/subset_900_with_split_exclude_cloud.ftr',             # Filepath for with split, exclude cloud (15)
    "fp_ns_ec"      : this_dir_fp + '/subset_900_no_split_exclude_cloud.ftr',               # Filepath for no split, exclude cloud (15)
    "fp_ws_core"    : this_dir_fp + '/subset_900_with_split_core.ftr',                      # Filepath for with split, core
    "fp_ws_core_sc" : this_dir_fp + '/subset_900_with_split_core_and_soft_core.ftr',        # Filepath for with split, core and soft core
    "fp_ws_core_10" : this_dir_fp + '/subset_900_with_split_core_10.ftr',                   # Filepath for with split, core to cloud 10
    "fp_ws_sh"      : this_dir_fp + '/subset_900_with_split_shell.ftr',                     # Filepath for with split, shell
    "fp_ws_sc_sh"   : this_dir_fp + '/subset_900_with_split_soft_core_and_shell.ftr',       # Filepath for with split, soft core and shell
    "fp_ws_total"   : this_dir_fp + '/subset_900_with_split_total.ftr',                     # Filepath for with split, total
    "fp_ws_shell_10": this_dir_fp + '/subset_900_with_split_shell_10.ftr'                   # Filepath for with split, shell to cloud 10
}
####################
####################


####################
### EXECUTION
####################
# Assign a name that will receive prefixes and suffixes on all outfiles
this_out_name = fp_to_out + "/" + designator + "_subset.ftr"
# Apply user input to select filepath
this_fp = dict_fp[designator]
# Create features and labels
X_data, y_data, unsplit_data = splitFeaturesLabels(this_fp, "asm_level", -400)
#showColStats(y_data, this_col)

# Copy feature and label dataframes for safety, subsetting all labels to interested labels here.
X_working = X_data.copy()
y_working = y_data[these_multilabels].copy()
X_working.reset_index(inplace=True)
X_working.to_feather(this_out_name)