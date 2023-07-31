#!/usr/bin/env python3

# Data handling
from math import comb
from random import sample
import numpy as np
import pandas as pd
from sklearn import preprocessing
# Searches
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
# Models
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.multioutput import MultiOutputClassifier
# Feature selection
from sklearn.feature_selection import chi2, SelectPercentile
# Metrics
from sklearn.metrics import accuracy_score, average_precision_score, auc, balanced_accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, fbeta_score, hamming_loss, hinge_loss, jaccard_score, log_loss, make_scorer, matthews_corrcoef, multilabel_confusion_matrix, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, top_k_accuracy_score, zero_one_loss
from sklearn.utils.multiclass import type_of_target
from sklearn.inspection import permutation_importance
# Visualization
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

#pip install imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import BalancedRandomForestClassifier

#pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# for dealing with multicollinearity
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
"""
From ml_testing_imbalearn.py

This script reads a combined feature and label matrix to do machine learning. Uses the "ml_imba_test_clone" environment.
Choose parameters from "02_gridsearch_hypers_rfc.py"

INSTALLATION:
- A fresh conda environment designated for python=3
-- pip3 install -U imbalanced-learn
-- conda install -c conda-forge imbalanced-learn
DO IT LIKE THIS OR THERE COULD BE SIGNIFICANT VERSIONING AND/OR LOCATING ERRORS
- check for scipy, numpy, scikit-learn
- then check on other dependencies like pandas, feather-format, keras, tensorflow, matplotlib, seaborn
- then need sampler
-- pip3 install iterative-stratification


PURPOSE:
- visualize model differences between:
    -- pangenome level choice
    -- resampling strategies
- show other useful visuals
    -- data sparsity
    -- donut plots

TODO:
    Modify the block of code so that all confusion matrix graphics can be in the same image (stacked vertically)
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

# Calculate the correlation for the dataset
def calculateCorrelationDataFrame( X_train, y_train, score_func_obj, score_func_name_str, percentile ):
    this_select = SelectPercentile(score_func=score_func_obj, percentile=percentile)
    this_select.fit(X_train, y_train)
    dict_this_select = dict(zip(X_train.columns, this_select.scores_))
    col_name_str = score_func_name_str + "_score"
    df_this_select = pd.DataFrame.from_dict(dict_this_select, orient="index", columns=[col_name_str])
    return(df_this_select)

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

# GridSearch CV result {'bootstrap': False, 'max_depth': 80, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 950}

# Filepath to results of feature importance calculations
fp_to_imp_results = '/scratch/dbrow208/galick_gun_working_dir/clean_run_20221130/fp_ws_shell_10_test_roc_auc_picked_RandomOverSampler_target_importance_values_per_features_roc_auc_TRAIN.tsv'
fp_to_imp2_results = '/scratch/dbrow208/galick_gun_working_dir/clean_run_20221130/fp_ws_shell_10_test_roc_auc_picked_RandomOverSampler_target_importance_values_per_features_roc_auc_VAL.tsv'
# Filepath to results of predictions for calculating AUC
fp_to_pred = '/scratch/dbrow208/galick_gun_working_dir/clean_run_20221130/fp_ws_shell_10_test_roc_auc_picked_RandomOverSampler_target_prediction_values_per_test_indices.tsv'
# Filepath to identify clusters
fp_to_clusters = '/scratch/dbrow208/galick_gun_working_dir/clean_run_20221130/fp_ws_shell_10_test_roc_auc_picked_RandomOverSampler_all_ranked_clusters_num_clust_66_roc_auc_VAL.tsv'

# Choose the designator for the input filepath (see below dictionary "dict_fp")
designator = 'fp_ws_shell_10'
# Set the filepath to the directory containing the feather format inputs
this_dir_fp = '/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests'
# Assign a filepath to the output directory
fp_to_graphic_dir = "/scratch/dbrow208/galick_gun_working_dir/clean_run_20221130"
# Assign other hyperparameters for the RandomForestClassifier
this_test_size = 0.15    # Float for percentage of samples as testing data set
this_rand_state = 1234  # For all random states
these_n_splits = 10      # For KFold
these_n_jobs = -1       # For all n_jobs, though 4 is also a good choice
num_chosen_genes = 20   # Integer for the number of chosen genes/features to include in plots and print outs
this_cut = 2
num_clusters = 66
choose_normal_fig = (12,8)
choose_bigger_fig = (18,12)
choose_biggest_fig = (24,16)
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
# Select a set of scoring metrics
scoring = ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_weighted']   # 'f1'
# Select the primary scoring metric for choosing an estimator
choose_score = '_roc_perm_imp'   # 'f1'
# Assign a name that will receive prefixes and suffixes on all outfiles
this_out_name = designator + choose_score
"""
# Possible alternate method.
scoring = {
    'accuracy'  :   make_scorer(balanced_accuracy_score),
    #'ck'        :   make_scorer(cohen_kappa_score),
    'cm'        :   make_scorer(confusion_matrix),
    #'hl'        :   make_scorer(hinge_loss),
    'mcc'       :   make_scorer(matthews_corrcoef),
    'roc_auc'   :   'roc_auc_ovo_weighted',
    #'roc_auc'   :   make_scorer(roc_auc_score, average='weighted', multi_class='ovo'),
    'top_k'     :   make_scorer(top_k_accuracy_score)

}
"""
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
# Apply user input to select filepath
this_fp = dict_fp[designator]
#####
### Data import
#####
# Instantiate output dataframe for metrics
df_out_full_estimator = pd.DataFrame()

# Report user inputs
print(
    "\nName", this_out_name,
    "\nFilepath", this_fp,
    "\nLabel Column", this_col,
    "\nMultilabel Columns", these_multilabels,
    "\nTest Size", this_test_size,
    "\nKFold Splits", these_n_splits,
    "\nChosen Metric", choose_score,
    "\nNum. Top Genes (features)", num_chosen_genes,
    "\n"
)

# Create features and labels
X_data, y_data, unsplit_data = splitFeaturesLabels(this_fp, "asm_level", -400)
#showColStats(y_data, this_col)

# Copy feature and label dataframes for safety, subsetting all labels to interested labels here.
X_working = X_data.copy()
y_working = y_data[these_multilabels].copy()


####################
### Make a two plot figure demonstrating multiple collinearity in the data set with a dendrogram and heat map
####################
# Please reference https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
# Set axes
fig1, (axA, axB) = plt.subplots(1, 2, figsize=choose_normal_fig)
# Calculate Spearman's correlation
corr = spearmanr(X_working).correlation
# Make the correlation matrix symmetrical
corr = (corr + corr.T) /2
np.fill_diagonal(corr, 1)
# Transform to a distance matrix
dist_matrix = 1 - np.abs(corr)  # Euclidean distance
# Calculate clusters
dist_link = hierarchy.ward(squareform(dist_matrix)) # Ward's linkage from the squareform Euclidean distance. Not the squared Euclidean distance. https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ward.html
# Determine selected features from a threshold cut
#cluster_ids = hierarchy.fcluster(dist_link, this_cut, criterion="distance")
cluster_ids = hierarchy.fcluster(dist_link, num_clusters, criterion='maxclust')    # Consider using once the "num_clusters" from the elbow plot is identified. See https://stackoverflow.com/questions/17616990/with-scipy-how-do-i-get-clustering-for-k-with-doing-hierarchical-clustering
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
#print(cluster_id_to_feature_ids)
selected_features = [ v[0] for v in cluster_id_to_feature_ids.values() ]

X_working_sel = X_working.iloc[:, selected_features]
### Visualize clusters with a dendrogram
dendro = hierarchy.dendrogram(
    dist_link, labels=X_working.columns.tolist(), ax=axB, leaf_rotation=90
)
# Create horizontal lines to represent potential thresholds
for i in range(1,5):
    if i == this_cut:
        axB.axhline(y=i, xmin=0, xmax=1, color='darkgrey')
    else:
        axB.axhline(y=i, xmin=0, xmax=1, color='darkgrey')
# Axis 2 - Labels and ticks off
axB.set_xticks([])
axB.set_xticklabels([])
axB.set_ylabel("Cophenetic Distance")

### Visualize the heatmap on the second axis
# Sets index to a list of leaves for the dendrogram
dendro_idx = np.arange(0, len(dendro["ivl"]))
axA.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
# Axis 1 - Labels and ticks on
#axA.set_xticks(dendro_idx)
#axA.set_yticks(dendro_idx)
#axA.set_xticklabels(dendro['ivl'], rotation='vertical')
#axA.set_yticklabels(dendro['ivl'])
# Axis 1 - Labels and ticks off
axA.set_xticks([])
axA.set_yticks([])
axA.set_xticklabels([])
axA.set_yticklabels([])
# Make title and save
fig1.suptitle("Feature Correlations Calculated using Spearman's Rank-order Correlation Coefficient\nand Resulting Feature Clusters from Ward's Linkage on the Euclidean Distance Matrix")
fig1.tight_layout()
out_dendro = fp_to_graphic_dir + "/" + this_out_name + "_dendrogram.png"
plt.savefig( out_dendro, format='png', dpi='figure', pad_inches=0.1 )
#plt.show()
plt.close()
###


####################
### Create a twinned axis scatterplot of number of clusters (y) vs cophenetic distance (x) and mean members per cluster (y) vs cophenetic distance (x) from the dendrogram above
####################
# Capture the limits of the previous dendrogram axis (distances)
cut_range = np.arange(axB.get_ylim()[0], axB.get_ylim()[1]+1, 1)
these_clusters = []
mean_members = []
for val in cut_range:
    assigned_clusters = hierarchy.fcluster(dist_link, val, criterion="distance")# Capture the maximum value of the flattened dendrogram at each distance
    these_clusters.append(np.amax(assigned_clusters))
    # Capture the count of members assigned to each unique cluster
    members = [ (assigned_clusters==i).sum() for i in np.unique(assigned_clusters) ]
    mean_members.append(np.mean(members))

fig2, axC = plt.subplots(figsize=choose_normal_fig, dpi=100)
#axD = axC.twinx()
axC.scatter(these_clusters, cut_range, c="darkorange", label="Number of Clusters")
axC.set_xscale('log')
axC.set_ylabel("Cophenetic Distance")
axC.set_xlabel("Log(10) Scale Values")
# Create horizontal lines to represent potential thresholds, as done on the dendrogram above
for i in range(1,5):
    if i == this_cut:
        axC.axhline(y=i, xmin=0, xmax=1, color='darkgrey')
    else:
        axC.axhline(y=i, xmin=0, xmax=1, color='darkgrey')
axC.scatter(mean_members, cut_range, c="royalblue", label="Mean Members per Cluster")
axC.axvline(x=66, ymin=0, ymax=0.85, color='crimson', label="Maximum Clusters of 66")
axC.set_xscale('log')
fig2.suptitle("Cophenetic Distance vs Log(10) Scale for Number of Clusters and Mean Members per Cluster for all Features")
fig2.tight_layout()
fig2.legend( loc="upper center", bbox_to_anchor=(0.5,0.99), bbox_transform=axC.transAxes )
out_scatter = fp_to_graphic_dir + "/" + this_out_name + "_scatter.png"
plt.savefig( out_scatter, format='png', dpi='figure', pad_inches=0.1 )
#plt.show()
plt.close()
###


# Read in dataframe
df_imp = pd.read_csv(fp_to_imp_results, sep='\t', index_col=0, low_memory=False)
# Copy for safety
df_imp_working = df_imp.copy()

dict_clusters = {}
df_clusters = pd.read_csv(fp_to_clusters, sep='\t', index_col=0, low_memory=False)
# Identify and format the single name for each cluster
cluster_names = df_clusters['aminoglycosides'].tolist()
cluster_names = [ pair.split(", ")[0] for pair in cluster_names ]
cluster_names = [ name.lstrip("('") for name in cluster_names ]
cluster_names = [ name.rstrip("'") for name in cluster_names ]
# Identify and format the contents for each cluster
cluster_contents = df_clusters['aminoglycosides_contents'].tolist()
cluster_contents = [ item.replace("'", "") for item in cluster_contents ]
cluster_contents = [ item.replace("[", "") for item in cluster_contents ]
cluster_contents = [ item.replace("]", "") for item in cluster_contents ]
cluster_contents = [ item.split(", ") for item in cluster_contents ]
for val in range(0, len(cluster_names)):
    for item in cluster_contents[val]:
        dict_clusters[item]= cluster_names[val]
print(dict_clusters)
####################
### Create the chi2 vs feature importance
####################
for i in these_multilabels:
    # Sort dataframe
    df_imp_working.sort_values(by=[i + "_feat_imp", "chi2_score"] , ascending=False, inplace=True)
    df_view_head = df_imp_working.head(num_chosen_genes)
    # Add in clusters containing the MutS and alt-name MutS genes for comparison
    df_view_head = pd.concat( [ df_imp_working[df_imp_working.index.str.startswith(dict_clusters['group_24522'])], df_view_head ] )
    df_view_head = pd.concat( [ df_imp_working[df_imp_working.index.str.startswith(dict_clusters['mutS'])], df_view_head ] )
    # Identify a range for the bar chart base axis (vertical bars == x)
    these_pos = np.arange(len(df_view_head.index.to_numpy().tolist()))
    # Set a bar width
    this_bar_width = 0.35
    fig3, axE = plt.subplots(figsize=choose_normal_fig, dpi=100)
    axF = axE.twinx()
    axE.bar(these_pos - this_bar_width/2, df_view_head["chi2_score"], this_bar_width, label="Chi-squared Score", color="darkgoldenrod")
    axE.set_ylabel("Chi-Squared Score")
    axF.bar(these_pos + this_bar_width/2, df_view_head[i + "_feat_imp"], this_bar_width, label="Mean Importance", color="lightseagreen")
    axF.set_ylabel("Mean Importance (Gini Impurity-based)")
    axE.set_xlabel("Selected Gene (feature)")
    axE.set_xticks( these_pos )
    axE.set_xticklabels( df_view_head.index.to_numpy().tolist(), rotation=45, ha="right" )
    fig3.suptitle("Training Set - Chi Squared and Impurity-based (Gini) Feature Importance for Selected Features\n" + i)
    fig3.tight_layout()
    fig3.legend( loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axF.transAxes )
    out_chi2_fi = fp_to_graphic_dir + "/" + this_out_name + "_" + i + "_train_chi2_fi.png"
    plt.savefig( out_chi2_fi, format='png', dpi='figure', pad_inches=0.1 )
    #plt.show()
    plt.close()
    ###

    ####################
    ### Create a two plot figure comparing a horizontal barchart of mean feature importance against box and whisker plots of permutation importance (sorted by mean feature importance)
    ####################
    # Please reference https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    # Sort dataframe
    df_imp_working.sort_values(by=[i + "_feat_imp", i + choose_score + "_mean"] , ascending=False, inplace=True)
    # Take top values
    df_view_head = df_imp_working.head(num_chosen_genes)
    df_view_head = df_view_head.copy()  # To avoid warnings about setting on a copy of a slice
    df_view_head.sort_values(by=[i + "_feat_imp", i + choose_score + "_mean"] , ascending=True, inplace=True)
    # Identify a range for the bar chart base axis (horizontal bars == y)
    these_pos = np.arange(len(df_view_head.index.to_numpy().tolist()))
    fig4, (axG, axH) = plt.subplots(1, 2, figsize=choose_normal_fig, dpi=100)
    axG.barh(these_pos, df_view_head[i + "_feat_imp"])
    axG.set_yticks(these_pos)
    axG.set_yticklabels(df_view_head.index.to_numpy().tolist())
    bad_type = df_view_head[i + choose_score + "_raw"].to_numpy().tolist() # As a list of strings
    bad_type = [ i.rstrip(']') for i in bad_type ]
    bad_type = [ i.lstrip('[') for i in bad_type ]
    bad_type = [ i.split(', ') for i in bad_type ]
    good_type = [ np.array(i, dtype=np.float32) for i in bad_type ]
    best_type = [i for i in good_type]
    final_transform = np.vstack(best_type)
    #print(df_view_head.index.to_numpy().tolist())
    axH.boxplot(
        final_transform.T,
        vert=False,
        labels=df_view_head.index.to_numpy().tolist(),
    )
    fig4.suptitle("Training Set - Bar Chart of Impurity-based (Gini) Feature Importances from Selected Features\nand Boxplot of Permutation Importances Sorted by Decreasing Feature Importance\n" + i)
    fig4.tight_layout()
    out_fi_pi = fp_to_graphic_dir + "/" + this_out_name + "_" + i + "_train_fi_pi_fi.png"
    plt.savefig( out_fi_pi, format='png', dpi='figure', pad_inches=0.1 )
    #plt.show()
    plt.close()

    ####################
    ### Create a two plot figure comparing a horizontal barchart of feature importance against box and whisker plots of permutation importance (sorted by mean permutation importance)
    ####################
    # Please reference https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    # Sort dataframe
    df_imp_working.sort_values(by=[i + choose_score + "_mean", i + "_feat_imp"] , ascending=False, inplace=True)
    # Take top values
    df_view_head = df_imp_working.head(num_chosen_genes)
    df_view_head = df_view_head.copy()  # To avoid warnings about setting on a copy of a slice
    df_view_head.sort_values(by=[i + choose_score + "_mean", i + "_feat_imp"] , ascending=True, inplace=True)
    # Identify a range for the bar chart base axis (horizontal bars == y)
    these_pos = np.arange(len(df_view_head.index.to_numpy().tolist()))
    fig5, (axI, axJ) = plt.subplots(1, 2, figsize=choose_normal_fig, dpi=100)
    axI.barh(these_pos, df_view_head[i + "_feat_imp"], color='crimson')
    axI.set_yticks(these_pos)
    axI.set_yticklabels(df_view_head.index.to_numpy().tolist())
    bad_type = df_view_head[i + choose_score + "_raw"].to_numpy().tolist() # As a list of strings
    bad_type = [ i.rstrip(']') for i in bad_type ]
    bad_type = [ i.lstrip('[') for i in bad_type ]
    bad_type = [ i.split(', ') for i in bad_type ]
    good_type = [ np.array(i, dtype=np.float32) for i in bad_type ]
    best_type = [i for i in good_type]
    final_transform = np.vstack(best_type)
    #print(df_view_head.index.to_numpy().tolist())
    axJ.boxplot(
        final_transform.T,
        vert=False,
        labels=df_view_head.index.to_numpy().tolist(),
    )
    fig5.suptitle("Training Set - Bar Chart of Impurity-based (Gini) Feature Importances from Selected Features\nand Boxplot of Permutation Importances Sorted by Decreasing Permutation Importance\n" + i)
    fig5.tight_layout()
    out_fi_pi = fp_to_graphic_dir + "/" + this_out_name + "_" + i + "_train_fi_pi_pi.png"
    plt.savefig( out_fi_pi, format='png', dpi='figure', pad_inches=0.1 )
    #plt.show()
    plt.close()

# Read in dataframe
df_imp2 = pd.read_csv(fp_to_imp2_results, sep='\t', index_col=0, low_memory=False)
# Copy for safety
df_imp_working = df_imp2.copy()

####################
### Create the chi2 vs feature importance
####################
for i in these_multilabels:
    # Sort dataframe
    df_imp_working.sort_values(by=[i + "_feat_imp", "chi2_score"] , ascending=False, inplace=True)
    df_view_head = df_imp_working.head(num_chosen_genes)
    # Add in clusters containing the MutS and alt-name MutS genes for comparison
    df_view_head = pd.concat( [ df_imp_working[df_imp_working.index.str.startswith(dict_clusters['group_24522'])], df_view_head ] ) # Check for alternate group name
    if df_imp_working.index.str.startswith(dict_clusters['mutS']) not in df_view_head.index.values.tolist():                        # Check for mutS
        df_view_head = pd.concat( [ df_imp_working[df_imp_working.index.str.startswith(dict_clusters['mutS'])], df_view_head ] )
    # Identify a range for the bar chart base axis (vertical bars == x)
    these_pos = np.arange(len(df_view_head.index.to_numpy().tolist()))
    # Set a bar width
    this_bar_width = 0.35
    fig3, axE = plt.subplots(figsize=choose_normal_fig, dpi=100)
    axF = axE.twinx()
    axE.bar(these_pos - this_bar_width/2, df_view_head["chi2_score"], this_bar_width, label="Chi-squared Score", color="darkgoldenrod")
    axE.set_ylabel("Chi-Squared Score")
    axF.bar(these_pos + this_bar_width/2, df_view_head[i + "_feat_imp"], this_bar_width, label="Mean Importance", color="lightseagreen")
    axF.set_ylabel("Mean Importance (Gini Impurity-based)")
    axE.set_xlabel("Selected Gene (feature)")
    axE.set_xticks( these_pos )
    axE.set_xticklabels( df_view_head.index.to_numpy().tolist(), rotation=45, ha="right" )
    fig3.suptitle("Validation Set - Chi Squared and Impurity-based (Gini) Feature Importance for Selected Features\n" + i)
    fig3.tight_layout()
    fig3.legend( loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axF.transAxes )
    out_chi2_fi = fp_to_graphic_dir + "/" + this_out_name + "_" + i + "_val_chi2_fi.png"
    plt.savefig( out_chi2_fi, format='png', dpi='figure', pad_inches=0.1 )
    #plt.show()
    plt.close()
    ###

    ####################
    ### Create a two plot figure comparing a horizontal barchart of feature importance against box and whisker plots of permutation importance (sorted by feature importance)
    ####################
    # Please reference https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    # Sort dataframe
    df_imp_working.sort_values(by=[i + "_feat_imp", i + choose_score + "_mean"] , ascending=False, inplace=True)
    # Take top values
    df_view_head = df_imp_working.head(num_chosen_genes)
    df_view_head = df_view_head.copy()  # To avoid warnings about setting on a copy of a slice
    df_view_head.sort_values(by=[i + "_feat_imp", i + choose_score + "_mean"] , ascending=True, inplace=True)
    # Identify a range for the bar chart base axis (horizontal bars == y)
    these_pos = np.arange(len(df_view_head.index.to_numpy().tolist()))
    fig4, (axG, axH) = plt.subplots(1, 2, figsize=choose_normal_fig, dpi=100)
    axG.barh(these_pos, df_view_head[i + "_feat_imp"])
    axG.set_yticks(these_pos)
    axG.set_yticklabels(df_view_head.index.to_numpy().tolist())
    bad_type = df_view_head[i + choose_score + "_raw"].to_numpy().tolist() # As a list of strings
    bad_type = [ i.rstrip(']') for i in bad_type ]
    bad_type = [ i.lstrip('[') for i in bad_type ]
    bad_type = [ i.split(', ') for i in bad_type ]
    good_type = [ np.array(i, dtype=np.float32) for i in bad_type ]
    best_type = [i for i in good_type]
    final_transform = np.vstack(best_type)
    #print(df_view_head.index.to_numpy().tolist())
    axH.boxplot(
        final_transform.T,
        vert=False,
        labels=df_view_head.index.to_numpy().tolist(),
    )
    fig4.suptitle("Validation Set - Bar Chart of Impurity-based (Gini) Feature Importances from Selected Features\nand Boxplot of Permutation Importances Sorted by Decreasing Feature Importance\n" + i)
    fig4.tight_layout()
    out_fi_pi = fp_to_graphic_dir + "/" + this_out_name + "_" + i + "_val_fi_pi_fi.png"
    plt.savefig( out_fi_pi, format='png', dpi='figure', pad_inches=0.1 )
    #plt.show()
    plt.close()

    ####################
    ### Create a two plot figure comparing a horizontal barchart of feature importance against box and whisker plots of permutation importance (sorted by permutation importance)
    ####################
    # Please reference https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    # Sort dataframe
    df_imp_working.sort_values(by=[i + choose_score + "_mean", i + "_feat_imp"] , ascending=False, inplace=True)
    # Take top values
    df_view_head = df_imp_working.head(num_chosen_genes)
    df_view_head = df_view_head.copy()  # To avoid warnings about setting on a copy of a slice
    df_view_head.sort_values(by=[i + choose_score + "_mean", i + "_feat_imp"] , ascending=True, inplace=True)
    # Identify a range for the bar chart base axis (horizontal bars == y)
    these_pos = np.arange(len(df_view_head.index.to_numpy().tolist()))
    fig5, (axI, axJ) = plt.subplots(1, 2, figsize=choose_normal_fig, dpi=100)
    axI.barh(these_pos, df_view_head[i + "_feat_imp"], color='crimson')
    axI.set_yticks(these_pos)
    axI.set_yticklabels(df_view_head.index.to_numpy().tolist())
    bad_type = df_view_head[i + choose_score + "_raw"].to_numpy().tolist() # As a list of strings
    bad_type = [ i.rstrip(']') for i in bad_type ]
    bad_type = [ i.lstrip('[') for i in bad_type ]
    bad_type = [ i.split(', ') for i in bad_type ]
    good_type = [ np.array(i, dtype=np.float32) for i in bad_type ]
    best_type = [i for i in good_type]
    final_transform = np.vstack(best_type)
    #print(df_view_head.index.to_numpy().tolist())
    axJ.boxplot(
        final_transform.T,
        vert=False,
        labels=df_view_head.index.to_numpy().tolist(),
    )
    fig5.suptitle("Validation Set - Bar Chart of Impurity-based (Gini) Feature Importances from Selected Features\nand Boxplot of Permutation Importances Sorted by Decreasing Permutation Importance\n" + i)
    fig5.tight_layout()
    out_fi_pi = fp_to_graphic_dir + "/" + this_out_name + "_" + i + "_val_fi_pi_pi.png"
    plt.savefig( out_fi_pi, format='png', dpi='figure', pad_inches=0.1 )
    #plt.show()
    plt.close()

####################
### Create a visualization of both Precision Recall and ROC for each individual classifier and that of the model as a whole
####################
# Read in dataframe
df_pred = pd.read_csv(fp_to_pred, sep='\t', index_col=0)
# Copy for safety
df_pred_working = df_pred.copy()
# Placeholder dataframe for positive prediction scores
df_pos_pred = pd.DataFrame()
# Instantiate lists to hold True Positive rates and AUC values
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# Plot the per classifier PR and ROC curves
fig6, (axK, axO) = plt.subplots(1, 2, figsize=choose_bigger_fig, dpi=100)
for i in these_multilabels:
    these_y_true = df_pred_working[i]
    these_y_pred = df_pred_working['pred_'+i]
    these_y_score = df_pred_working['prob_'+i].tolist()
    these_y_score = [ item.replace("'", "") for item in these_y_score ]
    these_y_score = [ item.replace("[", "") for item in these_y_score ]
    these_y_score = [ item.replace("]", "") for item in these_y_score ]
    these_y_score = [ float(item.split(", ")[1]) for item in these_y_score ]   # Take care to access the predicted probabilities for the positive score. - db 20221109
    df_pos_pred[i] = these_y_score
    prd = PrecisionRecallDisplay.from_predictions(
        these_y_true,
        these_y_score,
        name="Classifier {}".format(i),
        alpha=0.65,
        lw=1,
        ax=axK,
    )
    rcd = RocCurveDisplay.from_predictions(
        these_y_true,
        these_y_score,
        name="Classifier {}".format(i),
        alpha=0.65,
        lw=1,
        ax=axO,
    )
    interp_tpr = np.interp(mean_fpr, rcd.fpr, rcd.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(rcd.roc_auc)
"""
# Plot the full multilabel model PR and ROC curves ("multilabel-indicator format is not supported") - 20221111 db
model_prd = PrecisionRecallDisplay.from_predictions(
    df_pred_working[these_multilabels],
    df_pos_pred[these_multilabels],
    color='k',
    name="Multilabel Model",
    alpha=0.8,
    lw=2,
    ax=axK,
)
model_rcd = RocCurveDisplay.from_predictions(
    df_pred_working[these_multilabels],
    df_pos_pred[these_multilabels],
    color='k',
    name="Multilabel Model",
    alpha=0.8,
    lw=2,
    ax=axO,
)
"""
# Plot mean and standard deviation for the ROCs
#axO.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
axO.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean (all classifiers) (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
axO.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev. from Mean",
)
axK.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Precision Recall (PR) Curves with Calculated Average Precision (AP)",
)
axO.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver Operating Characteristic (ROC) Curves with Calculated Area Under ROC (AUC)",
)
axK.legend(loc="lower left")
axO.legend(loc="lower right")
fig6.suptitle("Evaluating per Target Binary Classifiers and Multilabel Model Predictions on the Testing Data")
fig6.tight_layout()
out_roc = fp_to_graphic_dir + "/" + this_out_name + "_pr_roc.png"
plt.savefig( out_roc, format='png', dpi='figure', pad_inches=0.1 )
#plt.show()
plt.close()

####################
### Create a visualization for an elbow criterion using k-means clustering on the SELECTED FEATURES (not samples)
####################
#X_t = X_working_sel.copy()
X_t = X_data.copy()
# Transpose to get features as rows
X_t = X_t.T
X_t.columns = [ str(i) for i in X_t.columns.tolist() ]
# Instantiate dictionary to hold sum of distances to cluster center
dict_inertia = {}
#for k in np.linspace(np.amin(these_clusters), np.amax(these_clusters)*0.95, 35):
for k in np.arange(15, 251, 1):
#for k in np.logspace(0.0, 3.7, num=15): # Default base is 10, start and stop values give beginning of sequence base**start (1) and end of sequence base**stop (~5000)
    k=int(k)    # Force type before passing to kmeans
    kmeans = KMeans(n_clusters=k, random_state=this_rand_state).fit(X_t)
    X_t['clusters'] = kmeans.labels_
    dict_inertia[k] = kmeans.inertia_
print(dict_inertia)
# Generate figure
fig7, axL = plt.subplots(figsize=choose_normal_fig, dpi=100)
axD = axL.twinx()
axL.plot(list(dict_inertia.keys()), list(dict_inertia.values()), c='purple', label="Elbow Criterion as Sum of Squared Errors")
axL.set_ylabel("Sum of Squared Errors (Distances to Closest Cluster Center)")
axL.set_xlabel("Number of Clusters")
axL.axvline(x=66, ymin=0, ymax=1, color='crimson', label="Maximum Clusters of 66")
axL.set_xlim(left=None, right=275)
axD.scatter(these_clusters, mean_members, c="royalblue", label="Mean Members per Cluster labelled with Cophenetic Distance")
axD.set_ylim(bottom=10, top=10000)
axD.set_yscale('log')
axD.set_ylabel("Mean Members per Cluster")
fig7.suptitle("Elbow Criterion for Sum of Squared Errors vs Number of Feature Clusters")
for i, txt in enumerate(cut_range.tolist()[:6]):
    axD.annotate( str(int(txt)), xy=(these_clusters[i], mean_members[i]), xytext=(these_clusters[i]+1, mean_members[i]+1))  # Offset the coordinates for the text. https://matplotlib.org/stable/gallery/pyplots/annotation_basic.html
fig7.tight_layout()
fig7.legend( loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=axL.transAxes )
out_elbow = fp_to_graphic_dir + "/" + this_out_name + "_elbow.png"
plt.savefig( out_elbow, format='png', dpi='figure', pad_inches=0.1 )
#plt.show()
plt.close()


####################
### Create a 2x5 graphic of box and whisker plots. Rows are TRAIN && VAL permutation importance, columns are each drug class target
####################
df_train = df_imp.copy()
df_val = df_imp2.copy()
fig8, axM = plt.subplots(2, len(these_multilabels), figsize=choose_biggest_fig, dpi=100)
fig8.tight_layout()
axM = axM.ravel()
these_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Upper row
these_patches = []
for target in these_multilabels:
    # Sort dataframe
    df_train.sort_values(by=[target + choose_score + "_median", target + "_feat_imp"] , ascending=False, inplace=True)
    # Take top values
    df_view_head = df_train.head(num_chosen_genes)
    df_view_head = df_view_head.copy()  # To avoid warnings about setting on a copy of a slice
    df_view_head.sort_values(by=[target + choose_score + "_median", target + "_feat_imp"] , ascending=True, inplace=True)
    bad_type = df_view_head[target + choose_score + "_raw"].to_numpy().tolist() # As a list of strings
    bad_type = [ i.rstrip(']') for i in bad_type ]
    bad_type = [ i.lstrip('[') for i in bad_type ]
    bad_type = [ i.split(', ') for i in bad_type ]
    good_type = [ np.array(i, dtype=np.float32) for i in bad_type ]
    best_type = [i for i in good_type]
    final_transform = np.vstack(best_type)
    #print(df_view_head.index.to_numpy().tolist())
    holder = axM[these_multilabels.index(target)].boxplot(
        final_transform.T,
        vert=False,
        labels=df_view_head.index.to_numpy().tolist(),
        patch_artist=True   # Enable color changes via patch artists
    )
    # Fill boxes with color https://matplotlib.org/stable/gallery/statistics/boxplot_color.html
    for patch in holder['boxes']:
        patch.set_facecolor(these_colors[these_multilabels.index(target)])
        patch.set_alpha(0.65)
    # Change the median line color
    for patch in holder['medians']:
        patch.set_color('k')
    # Format title
    axM[these_multilabels.index(target)].set_title(target)
    # Format labels
    if target == these_multilabels[0]:
        axM[these_multilabels.index(target)].set_ylabel('Training Data Set')
# Lower row
for target in these_multilabels:
    # Sort dataframe
    df_val.sort_values(by=[target + choose_score + "_median", target + "_feat_imp"] , ascending=False, inplace=True)
    # Take top values
    df_view_head = df_val.head(num_chosen_genes)
    df_view_head = df_view_head.copy()  # To avoid warnings about setting on a copy of a slice
    df_view_head.sort_values(by=[target + choose_score + "_median", target + "_feat_imp"] , ascending=True, inplace=True)
    bad_type = df_view_head[target + choose_score + "_raw"].to_numpy().tolist() # As a list of strings
    bad_type = [ i.rstrip(']') for i in bad_type ]
    bad_type = [ i.lstrip('[') for i in bad_type ]
    bad_type = [ i.split(', ') for i in bad_type ]
    good_type = [ np.array(i, dtype=np.float32) for i in bad_type ]
    best_type = [i for i in good_type]
    final_transform = np.vstack(best_type)
    #print(df_view_head.index.to_numpy().tolist())
    holder = axM[these_multilabels.index(target)+len(these_multilabels)].boxplot(
        final_transform.T,
        vert=False,
        labels=df_view_head.index.to_numpy().tolist(),
        patch_artist=True   # Enable color changes via patch artists
    )
    # Fill boxes with color https://matplotlib.org/stable/gallery/statistics/boxplot_color.html
    for patch in holder['boxes']:
        patch.set_facecolor(these_colors[these_multilabels.index(target)])
        patch.set_alpha(0.5)
    # Change the median line color
    for patch in holder['medians']:
        patch.set_color('k')
    # Format labels
    if target == these_multilabels[0]:
        axM[these_multilabels.index(target)+len(these_multilabels)].set_ylabel('Validation Data Set')
# Add gridlines
for ax in axM:
    ax.xaxis.grid(True, color='lightgray')

fig8.suptitle("Comparing Boxplots of Permutation Importances from Training and Validation Sets\nSorted by Decreasing Feature Importance")
#fig8.subplots_adjust(wspace=0.4, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)
fig8.tight_layout()
out_all_boxes = fp_to_graphic_dir + "/" + this_out_name + "_all_pi_boxes.png"
plt.savefig( out_all_boxes, format='png', dpi='figure', pad_inches=0.1 )
#plt.show()
plt.close()
