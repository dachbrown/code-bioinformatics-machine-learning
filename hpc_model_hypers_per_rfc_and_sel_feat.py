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

#pip install imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
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
### FILEPATHS TO DATASETS
####################
# All filepaths
dict_fp = {
    "fp_ws_ec"      : '/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/subset_900_with_split_exclude_cloud.ftr',            # Filepath for with split, exclude cloud (15)
    "fp_ns_ec"      : '/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/subset_900_no_split_exclude_cloud.ftr',              # Filepath for no split, exclude cloud (15)
    "fp_ws_core"    : '/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/subset_900_with_split_core.ftr',                   # Filepath for with split, core
    "fp_ws_core_sc" : '/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/subset_900_with_split_core_and_soft_core.ftr',  # Filepath for with split, core and soft core
    "fp_ws_core_10" : '/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/subset_900_with_split_core_10.ftr',             # Filepath for with split, core to cloud 10
    "fp_ws_sh"      : '/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/subset_900_with_split_shell.ftr',                    # Filepath for with split, shell
    "fp_ws_sc_sh"   : '/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/subset_900_with_split_soft_core_and_shell.ftr',   # Filepath for with split, soft core and shell
    "fp_ws_total"   : '/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/subset_900_with_split_total.ftr',                 # Filepath for with split, total
    "fp_ws_shell_10": '/scratch/dbrow208/galick_gun_working_dir/20221019_ml_tests/subset_900_with_split_shell_10.ftr'         # Filepath for with split, shell to cloud 10
}
# A testing subset
dict_single = {
    "fp_ns_ec" : '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_no_split_exclude_cloud.ftr',              # Filepath for no split, exclude cloud (15)
}
####################
####################


####################
### USER INPUTS - for specific parameters
####################

# GridSearch CV result {'bootstrap': False, 'max_depth': 80, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 950}

# Choose the input filepaths to investigate
designator = 'fp_ws_shell_10'
this_fp = dict_fp[designator]
# Assign a filepath to the output directory
fp_to_graphic_dir = "/scratch/dbrow208/galick_gun_working_dir/clean_run_20221130"
# Assign other hyperparameters for the RandomForestClassifier
this_test_size = 0.15    # Float for percentage of samples as testing data set
this_rand_state = 1234  # For all random states
these_n_splits = 10     # For KFold
these_n_jobs = -1       # For all n_jobs, though 4 is also a good choice
num_chosen_genes = 20   # Integer for the number of chosen genes/features to include in plots and print outs
these_n_repeats = 100    # For permutation importance
this_cut = 2            # Choose cut for cophenetic distance when selecting clusters
num_clusters = 66        # Choose number of clusters from dendrogram (taken from an elbow plot)
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
#scoring = ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_weighted']   # For the 
# Select the scoring metric for choosing an estimator in the training phase (reference dictionary "scoring")
choose_score = 'test_roc_auc'   # Need "test_" prefix here, this is for the combined multilabel classifier. - db 20221107
# Select the scoring metric for choosing an estimator in the validation phase while calculating permutation importance (reference dictionary "scoring")
pick_method = 'roc_auc'    # DO NOT NEED PREFIX HERE, could change the formula from 'weighted to 'binary' for each individual binary classifier during permutation importance. - db 20221107
# Assign a name that will receive prefixes and suffixes on all outfiles
this_out_name = designator + "_" + choose_score

# Possible alternate method for a binary (multi)class. (Not multilabel)
scoring = {
    'accuracy'          : 'accuracy',
    'balanced_accuracy' : 'balanced_accuracy',
    'f1'                : 'f1',
    'f1_weighted'       : 'f1_weighted',
    'mcc'               : make_scorer(matthews_corrcoef),
    'roc_auc'           : 'roc_auc',
}

# NOTE The imblearn BRFC will not use multilabel targets, so you can hack around it by piping only one target at a time, or employ the RFC from scikit-learn.
# Either RandomForestClassifier is useful. As of 20221019, the default settings on both appear to be the same.
# From scikit-learn
"""
# From a first run of random -> grid, based on ws_shell, with potentially flawed
rfc = RandomForestClassifier(
    n_estimators=these_n_estimators,
    #oob_score=True,    # Only available if bootstrap=True
    bootstrap=False,
    max_depth=80,
    max_features=3,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=this_rand_state,
    n_jobs=these_n_jobs,
    class_weight="balanced"
)
# Second round of random -> grid, based on ws_shell
rfc = RandomForestClassifier(
    n_estimators=these_n_estimators,
    #oob_score=True,    # Only available if bootstrap=True
    bootstrap=False,
    max_depth=25,
    max_features=4,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=this_rand_state,
    n_jobs=these_n_jobs,
    class_weight="balanced"
)
""" # Classifiers fit on cophenetic distance 1 data, chosen by maximizing && refitting on the ROC AUC during the randomized (x1000) cross validation search.
dict_rfc = {
    "aminoglycosides" :
        RandomForestClassifier(
            n_estimators=800,
            min_samples_split=20,
            min_samples_leaf=4,
            max_features='log2',
            max_depth=50,
            random_state=this_rand_state,
            n_jobs=these_n_jobs,
            bootstrap=True,
        ),
    "folate_pathway_antagonists" :
        RandomForestClassifier(
            n_estimators=725,
            min_samples_split=20,
            min_samples_leaf=8,
            max_features='log2',
            max_depth=70,
            random_state=this_rand_state,
            n_jobs=these_n_jobs,
            bootstrap=False,
        ),
    "macrolides" :
        RandomForestClassifier(
            n_estimators=175,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            max_depth=30,
            random_state=this_rand_state,
            n_jobs=these_n_jobs,
            bootstrap=False,
        ),
    "tetracyclines" :
        RandomForestClassifier(
            n_estimators=375,
            min_samples_split=2,
            min_samples_leaf=8,
            max_features=0.2,
            max_depth=40,
            random_state=this_rand_state,
            n_jobs=these_n_jobs,
            bootstrap=True,
        ),
    "other" :
        RandomForestClassifier(
            n_estimators=350,
            min_samples_split=12,
            min_samples_leaf=2,
            max_features=0.1,
            max_depth=100,
            random_state=this_rand_state,
            n_jobs=these_n_jobs,
            bootstrap=False,
        ),
}
# {'bootstrap': False, 'max_depth': 25, 'max_features': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1380}
# From imblearn
#rfc = BalancedRandomForestClassifier(n_estimators=these_n_estimators, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
####################
####################


####################
### EXECUTION
####################

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
    "\nCophenetic Distance Cut", this_cut,
    "\nNum. Clusters", num_clusters,
    "\nNum. Top Genes (features)", num_chosen_genes,
    "\nNum. Repeats (PI)", these_n_repeats,
    "\n"
)

# Create features and labels
X_data, y_data, unsplit_data = splitFeaturesLabels(this_fp, "asm_level", -400)
#showColStats(y_data, this_col)

# Copy feature and label dataframes for safety, subsetting all labels to interested labels here.
X_working = X_data.copy()
y_working = y_data[these_multilabels].copy()

#####
### Handling collinearity
#####
# Please reference https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
corr = spearmanr(X_working).correlation
corr = (corr + corr.T) /2
np.fill_diagonal(corr, 1)
dist_matrix = 1 - np.abs(corr)
# Calculate clusters
dist_link = hierarchy.ward(squareform(dist_matrix)) # Euclidean distance, not square from https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ward.html

# Determine selected features from a threshold cut
#cluster_ids = hierarchy.fcluster(dist_link, this_cut, criterion="distance")
cluster_ids = hierarchy.fcluster(dist_link, num_clusters, criterion='maxclust')    # Consider using once the "num_clusters" from the elbow plot is identified. See https://stackoverflow.com/questions/17616990/with-scipy-how-do-i-get-clustering-for-k-with-doing-hierarchical-clustering
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
#print(cluster_id_to_feature_ids)
selected_features = [ v[0] for v in cluster_id_to_feature_ids.values() ]

X_working_sel = X_working.iloc[:, selected_features]

####################
### SEPARATING SELECTED FEATURES INTO TESTING, TRAINING, AND VALIDATION SETS
####################
# Determine method for splitting into training/validation and testing data sets
msss_test = MultilabelStratifiedShuffleSplit( n_splits=1, test_size=this_test_size, random_state=this_rand_state )
# Determine method for splitting into training and validation data sets
msss_val = MultilabelStratifiedShuffleSplit( n_splits=1, test_size=this_test_size/(1-this_test_size), random_state=this_rand_state )    # db 20230130 - the "this_test_size/(1-this_test_size)" correction is needed to arrive at equal sized (count) training/validation sets when passing a float (percentage) as a split parameter

# Explicitly create empty dataframes for safety
# 'train'   -- stratified equivalent to the original data, whole training set is used for tuning hyperparameters on each classifier, will later be resampled on a per classifier basis when training
# 'val'     -- stratified equivalent to the original data, used for validating the hyperparameters by calculating errors, will not be resampled, will later be used to calculate permutation importance
# 'test'    -- stratified equivalent to the original data, not used here, will not be resampled, will later be used for testing the final model (composite of classifiers)
X_train_val = pd.DataFrame()    # Training and validation features
X_test = pd.DataFrame()         # Testing features for final model test
X_train = pd.DataFrame()        # Training features for learning and hyperparameter tuning
X_val = pd.DataFrame()          # Validation features for permutation importance
y_train_val = pd.DataFrame()    # Training and validation features
y_test = pd.DataFrame()         # Testing features for final model test
y_train = pd.DataFrame()        # Training features for learning and hyperparameter tuning
y_val = pd.DataFrame()          # Validation features for permutation importance

# Split & stratify the ENTIRE into training/validation and testing sets to prevent data leakage and preserve real-world label ratios for final testing of the composite model.
for train_index, test_index in msss_test.split(X_working_sel, y_working):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train_val, X_test = X_working_sel.iloc[train_index], X_working_sel.iloc[test_index]
    y_train_val, y_test = y_working.iloc[train_index], y_working.iloc[test_index]

# Split & stratify into training and validation sets to prevent data leakage and preserve real-world label ratios for tuning the hyperparameters of each binary classifier in the composite model.
for train_index, test_index in msss_val.split(X_train_val, y_train_val):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[test_index]
    y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[test_index]

print("Compare the below values to the above.\nAfter feature selection by reducing collinear features,")
print("Feature size of the blinded testing set", X_test.shape)
print("Feature size of the validation set", X_val.shape)
print("Feature size of the training set", X_train.shape)

### Determine sampling methods
# Undersampling of the majority class
us_rnd = RandomUnderSampler( sampling_strategy='not minority', random_state=this_rand_state )
us_enn = EditedNearestNeighbours( sampling_strategy='not minority', kind_sel='mode', n_jobs=these_n_jobs )
# Oversampling of the minority class
os_rnd = RandomOverSampler( sampling_strategy='not majority', random_state=this_rand_state )
os_smote = SMOTE ( sampling_strategy='not majority', random_state=this_rand_state, n_jobs=these_n_jobs )
# Combination of resampling
combo_smoteenn = SMOTEENN( random_state=this_rand_state, n_jobs=these_n_jobs )
combo_smotetmk = SMOTETomek( random_state=this_rand_state, n_jobs=these_n_jobs )
sample_methods_list = [ us_rnd, os_rnd, os_smote, combo_smoteenn, combo_smotetmk ]
#sample_methods_list = [ us_rnd, os_rnd, combo_smoteenn ]

# Note that imblearn does not allow 'multi-label' indicator data, so we are getting around that by treating each target column individually
mskf = MultilabelStratifiedKFold ( n_splits=these_n_splits, shuffle=True, random_state=this_rand_state )
these_mskf_splits = list(mskf.split(X_train, y_train))    # This is required here, as a multilabel-indicator is required. The splits are passed to a later cv= within cross_validate


"""
According to sklearn the multioutputclassifier "This strategy consists of fitting one classifier per target. This is a simple strategy for extending classifiers that do not natively support multi-target classification." https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier
I guess the multioutputclassifier just stacks (hstacks?) the prediction && predict_proba, before passing to the scoring metrics
I WILL DO THE SAME, BUT BREAK APART THE IMPLEMENTATION.
    The main issue then becomes cross-validation, as the oversamplings in each classifier will not be the same.
        e.g. To arrive at oversampled tet sequences you would likely not use the oversampled phenicol indices.

so
    msss split the dataset (via multilabel stratified shuffle split to preserve label ratios in wild)
    then choose the validation indices using mskf on the training set (multilabelstratifiedkfolds)
    # NOTE that folds are held constant for all targets, stratifiedkfolds are selected from stratified & shuffled training set, then resampled in various ways
    then for each fold:  
        then for each target (column):
            oversample the minority
            fit classifier
            make predictions (for that target)
            filter predictions for median classifier for that target
        identify multilabel scores from the grouped predictions of that median classifier for each target, focusing on accuracy, precision, recall, f1, HammingLoss
        store the scores - WIP
    results in (folds*targets) number of estimators - NO, could do this, but just reporting the multilabel metrics on the results of the best scoring clasifier per target
    compare the scores (multilabel per fold && each target score internal range)
"""
#####
### Perform cross-validation, collect metrics, and visualize
#####
sampler_scores = []
dict_best_estimators = {}
#dict_sampler['y_test'] = y_test
# Run pipeline and collect metrics
for sampler in sample_methods_list:
    #model = make_pipeline(sampler, rfc)
    dict_targets_cv = {}
    all_y_pred = [] # Just a placeholder, as empty nparrays have constraints
    all_y_score = [] # Just a placeholder, as empty nparrays have constraints
    # Perform cross validation while fitting the estimators, results are stored by target
    for target in y_train.columns.tolist():
        model = make_pipeline(sampler, dict_rfc[target])
        cv_return = cross_validate( model, X_train, y_train[target], cv=these_mskf_splits, scoring=scoring, n_jobs=these_n_jobs, return_estimator=True )
        dict_targets_cv[target] = cv_return
        print('\n\nFor', target)
        choose_fold = 0
        # Calculate feature importances per fold
        print('\nFeature Importances by Gini Impurity')
        for fold in range(len(cv_return['estimator'])):
            these_fi = list( zip( X_train.columns, cv_return['estimator'][fold][-1].feature_importances_ ) ) 
            these_fi.sort( reverse=True, key=lambda x: x[1] )
            just_genes = [ pair[0] for pair in these_fi ]
            print("fold", fold, these_fi[:num_chosen_genes])
            #print("fold", fold, just_genes[:num_chosen_genes])
        # Choose one estimator per target, selected from the median of all folds for "choose_score"
        fold_median = ( np.abs( cv_return[choose_score] - np.median(cv_return[choose_score]) ) ).argmin()   # The index location of the element closest to the median has the shortest absolute distance
        fold_max = np.argmax(cv_return[choose_score])    
        print('\nFor', target)
        choose_fold = fold_median
        print( "\nThe median value for ", choose_score, " is ", str(np.median(cv_return[choose_score])), " and the closest estimator score to that median value is ", str(cv_return[choose_score][fold_median]), " from fold ", str(fold_median), "\n" )
        choose_fold = fold_max
        print( "\nThe max value for ", choose_score, " is ", str(np.amax(cv_return[choose_score])), " from fold ", str(fold_max), "\n" )
        #print(cv_return)
        for score in cv_return.keys():
            if score != 'estimator':
                this_mean = np.mean(cv_return[score], axis=0)
                this_std = np.std(cv_return[score])
                print( score, "mean: ", this_mean, " +/- ", this_std )
            else:
                print("The following OOB should be similar to the mean scores")
                try:
                    print("OOB", cv_return['estimator'][choose_fold][-1].oob_score_)
                except:
                    print("Possible Warning. No OOB score (classifier.oob_score_), bootstraps are likely set to False")
        # Assign to dictionary
        dict_targets_cv[target]['best_estimator'] = cv_return['estimator'][choose_fold][-1] # Has already been fitted.
        #dict_best_estimators[str(sampler).split('(')[0]] = { target : cv_return['estimator'][choose_fold][-1] }
        dict_targets_cv[target]['best_predictions'] = cv_return['estimator'][choose_fold][-1].predict(X_test)
        dict_targets_cv[target]['best_scores'] = cv_return['estimator'][choose_fold][-1].predict_proba(X_test)
        # Report the classification report from imblearn
        print('\nimblearn Binary Classification Report for', sampler, "on", target)
        #print('\nFor', target)
        print(classification_report_imbalanced(y_test[target], dict_targets_cv[target]['best_predictions'], zero_division=0))
        # Access the pipeline object for the estimator to report feature importances
        # cross_validate returns a dictionary, requiring a the 'estimator' key to access all the returned, fitted estimators. Each estimator is accessed via integer for the CV fold step. When a pipeline is used, the desired estimator is found via [integer][-1][1] to represent the final step of the pipeline and the estimator
        #print(sorted(cv_return['estimator'][choose_fold][-1].feature_importances_, reverse=True, key=lambda x: x[1])[:num_chosen_genes])
        #print('see fold genes above for fold ', choose_fold)
        #print( dict_targets_cv[target])
        print("****")
    # Stack the predictions and scores from the best fold classifier on a per target basis PSEUDOCODE np.hstack(fold0 target0, fold2 target1, etc.)
    # Process data according to https://scikit-learn.org/stable/modules/model_evaluation.html#multi-label-case
    dict_best_per_sampler = {}
    for key in dict_targets_cv.keys():
        dict_best_per_sampler[key] = dict_targets_cv[key]['best_estimator']
        all_y_score.append(dict_targets_cv[key]['best_scores'])
        if list(dict_targets_cv.keys()).index(key) == 0:
            all_y_pred = np.transpose([dict_targets_cv[key]['best_predictions']])
        else:
            all_y_pred = np.hstack( (all_y_pred, np.transpose([dict_targets_cv[key]['best_predictions']])) )
    all_y_score = np.transpose([y_pred[:, 1] for y_pred in all_y_score])
    dict_best_estimators[str(sampler).split('(')[0]] = dict_best_per_sampler
    #[str(sampler).split('(')[0]]['all_y_score'] = all_y_score
    #dict_sampler[str(sampler).split('(')[0]]['all_y_preds'] = all_y_pred

    # Visuals
    # Please reference https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
    macro_auc_roc = roc_auc_score(y_test, all_y_score, average='macro' )    # Calculated here to include in a file name. Enables visual comparison in a file directory for relative value of predictions.
    fig, axes = plt.subplots(4, len(these_multilabels), figsize=(15, 10))
    fig.tight_layout()
    axes = axes.ravel()
    fig.gca().tick_params(color='w')
    for i in range(len(these_multilabels)):
        this_sub = ConfusionMatrixDisplay.from_predictions(
            y_test.iloc[:,i],
            all_y_pred[:,i],
            display_labels=[0,1],
            #normalize='true'
        )
        this_sub.plot(ax=axes[i])#, values_format='.4g')
        this_sub.ax_.set_title(y_test.columns.tolist()[i])
        this_sub.ax_.set_xlabel('')
        if i!=0:
            this_sub.ax_.set_ylabel('')
        else:
            this_sub.ax_.set_ylabel('Not Normalized\nTrue Label')
        this_sub.im_.colorbar.remove()
        plt.close()
    for i in range(len(these_multilabels)):
        this_sub = ConfusionMatrixDisplay.from_predictions(
            y_test.iloc[:,i],
            all_y_pred[:,i],
            display_labels=[0,1],
            normalize='true',
            values_format='.0%'
        )
        this_sub.plot(ax=axes[i+len(these_multilabels)])#, values_format='.4g')
        #this_sub.ax_.set_title(y_test.columns.tolist()[i])
        this_sub.ax_.set_xlabel('')
        if i!=0:
            this_sub.ax_.set_ylabel('')
        else:
            this_sub.ax_.set_ylabel('True Normalized\nTrue Label')
        this_sub.im_.colorbar.remove()
        plt.close()
    for i in range(len(these_multilabels)):
        this_sub = ConfusionMatrixDisplay.from_predictions(
            y_test.iloc[:,i],
            all_y_pred[:,i],
            display_labels=[0,1],
            normalize='pred',
            values_format='.0%'
        )
        this_sub.plot(ax=axes[i+(2*len(these_multilabels))])#, values_format='.4g')
        #this_sub.ax_.set_title(y_test.columns.tolist()[i])
        this_sub.ax_.set_xlabel('')
        if i!=0:
            this_sub.ax_.set_ylabel('')
        else:
            this_sub.ax_.set_ylabel('Prediction Normalized\nTrue Label')
        this_sub.im_.colorbar.remove()
        plt.close()
    for i in range(len(these_multilabels)):
        this_sub = ConfusionMatrixDisplay.from_predictions(
            y_test.iloc[:,i],
            all_y_pred[:,i],
            display_labels=[0,1],
            normalize='all',
            values_format='.0%'
        )
        this_sub.plot(ax=axes[i+(3*len(these_multilabels))])#, values_format='.4g')
        #this_sub.ax_.set_title(y_test.columns.tolist()[i])
        if i!=0:
            this_sub.ax_.set_ylabel('')
        else:
            this_sub.ax_.set_ylabel('All Normalized\nTrue Label')
        this_sub.im_.colorbar.remove()
        plt.close()
    #fig.suptitle( str(sampler) + " MultiLabel " + this_fp )
    fig.suptitle( str(sampler).split('(')[0] + "- BRFC - MultiLabel - Macro ROC AUC " + str(round(macro_auc_roc, 3)) + " " + this_fp )
    # Add colorbar back in
    #fig.colorbar(this_sub.im_, ax=axes)
    #fig.colorbar(ax=axes, ticks=[0,1]).set_ticklabels(['Min', 'Max'])#, plt.tick_params=).set_ticks([0,1])
    #fig.colorbar(this_sub.im_, ax=axes).set_ticklabels(['Min','Max'])
    # Adjustments when including colorbar
    #fig.subplots_adjust(wspace=0.3, hspace=0.4, left=0.05, top=0.9, bottom=0.1)
    # Adjustments when NOT including any colorbar
    fig.subplots_adjust(wspace=0.4, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)
    out_fp = fp_to_graphic_dir + "/" + this_out_name + "_" + str(sampler).split('(')[0] + "_macro_auroc_" + str(round(macro_auc_roc, 3)) + ".png"
    plt.savefig( out_fp, format='png', dpi='figure', pad_inches=0.1 )
    #plt.show()
    plt.close()

    print(sampler, " Full MultiLabel Metrics - Classification Report")
    # NOTE "multilabel_confusion_matrix" from scikit-learn is very very broken, potentially due to numpy/numba/Python conflicts - 20221019 db
    print( classification_report(y_test, all_y_pred, zero_division=0, target_names=y_test.columns.tolist()) )
    
    dict_scores = {
        'test_accuracy_score'                           :   accuracy_score(y_test, all_y_pred),
        #'test_f1_binary'                                :   f1_score(y_test, all_y_pred, average='binary'),    # Cannot use when testing the Multilabel metrics as a whole, but can use on each individual classifier. - db 20221107
        'test_f1_weighted'                              :   f1_score(y_test, all_y_pred, average='weighted'),
        'test_f1_macro'                                 :   f1_score(y_test, all_y_pred, average='macro'),
        'test_f1_micro'                                 :   f1_score(y_test, all_y_pred, average='micro'),
        'test_fbeta_score_betadot5_avg_weighted'        :   fbeta_score(y_test, all_y_pred, beta=0.5, average='weighted'),
        'test_fbeta_score_betadot5_avg_macro'           :   fbeta_score(y_test, all_y_pred, beta=0.5, average='macro'),
        'test_fbeta_score_betadot5_avg_micro'           :   fbeta_score(y_test, all_y_pred, beta=0.5, average='micro'),
        'test_hamming_loss'                             :   hamming_loss(y_test, all_y_pred),
        'test_jaccard_score_weighted'                   :   jaccard_score(y_test, all_y_pred, average='weighted'),
        'test_jaccard_score_macro'                      :   jaccard_score(y_test, all_y_pred, average='macro'),
        'test_jaccard_score_micro'                      :   jaccard_score(y_test, all_y_pred, average='micro'),
        'test_log_loss'                                 :   log_loss(y_test, all_y_pred),
        'test_precision_recall_fscore_support_none'     :   precision_recall_fscore_support(y_test, all_y_pred),
        'test_precision_recall_fscore_support_weighted' :   precision_recall_fscore_support(y_test, all_y_pred, average='weighted' ),
        'test_precision_recall_fscore_support_macro'    :   precision_recall_fscore_support(y_test, all_y_pred, average='macro' ),
        'test_precision_recall_fscore_support_micro'    :   precision_recall_fscore_support(y_test, all_y_pred, average='micro' ),
        'test_roc_auc'                                  :   roc_auc_score(y_test, all_y_score),
        'test_roc_auc_weighted'                         :   roc_auc_score(y_test, all_y_score, average='weighted' ),
        'test_roc_auc_macro'                            :   roc_auc_score(y_test, all_y_score, average='macro' ),
        'test_roc_auc_micro'                            :   roc_auc_score(y_test, all_y_score, average='micro' ),
        'test_zero_one_loss'                            :   zero_one_loss(y_test, all_y_pred, normalize=True),
        'test_average_precision_score'                  :   average_precision_score(y_test, all_y_score),
        'density_ratio'                                 :   float(X_train.astype(pd.SparseDtype("int", 0)).sparse.density),
        'sparsity_ratio'                                :   1-float(X_train.astype(pd.SparseDtype("int", 0)).sparse.density)
    }

    sampler_scores.append(dict_scores[choose_score])
    #for score in dict_scores.keys():
        #print("\n", score)
        #print(dict_scores[score])
    
    col_name = designator + "_" + str(sampler).split('(')[0]
    df_out_full_estimator[col_name] = [ dict_scores[score] for score in dict_scores.keys() ]
    df_out_full_estimator['scores'] = list(dict_scores.keys())
    df_out_full_estimator.set_index( 'scores', inplace=True )
    
#####
### Output results of cross-validation and collected metrics
#####
# Save and output metrics
df_out_full_estimator.reset_index(inplace=True)
outname_df_out_full_estimator = fp_to_graphic_dir + "/" + this_out_name + "_df_all_sampler_subset_scores_combined_estimator.tsv"
df_out_full_estimator.to_csv(outname_df_out_full_estimator, sep='\t')
#[str(sampler).split('(')[0]]['targets_dict'] = dict_targets_cv
print(sampler_scores)
this_best_sampler = str( sample_methods_list[sampler_scores.index( max( sampler_scores ) )] ).split('(')[0]
print("The best sampler is:", this_best_sampler)

# FROM TRAINING - Capture the predictions and test values for the full model as a checkpoint/sanity check. For checking in the future.
df_labels = y_test.copy()
for key in dict_best_estimators[this_best_sampler].keys():
    these_train_y_pred = dict_best_estimators[this_best_sampler][key].predict(X_train)
    these_y_pred = dict_best_estimators[this_best_sampler][key].predict(X_test)
    these_y_prob = dict_best_estimators[this_best_sampler][key].predict_proba(X_test).tolist()
    name_y_pred = "pred_" + key
    name_y_prob = "prob_" + key
    df_labels[name_y_pred] = these_y_pred
    df_labels[name_y_prob] = these_y_prob
    print('\nThe best estimator for:\n', key, dict_best_estimators[this_best_sampler][key])
    print('The below scores should be similar:')
    print('training accuracy', dict_best_estimators[this_best_sampler][key].score(X_train, y_train[key]))
    print('testing accuracy', dict_best_estimators[this_best_sampler][key].score(X_test, y_test[key]))
    print('training accuracy balanced', balanced_accuracy_score(y_train[key], these_train_y_pred) )
    print('testing accuracy balanced', balanced_accuracy_score(y_test[key], these_y_pred) )
outname_df_labels = fp_to_graphic_dir + "/" + this_out_name + "_picked_" + this_best_sampler + "_target_prediction_values_per_test_indices.tsv"
df_labels.to_csv( outname_df_labels, sep='\t')


# FROM TRAINING - Capture the chi2, Gini impurity based feature importances, and permutation importances of the features
df_final_out = calculateCorrelationDataFrame( X_train, y_train, chi2, "chi2", 100 )
print('\nFeature Importances from Final Model by Permutation Importance on Train Data')
dict_target_top = {}
for target in y_train.columns.tolist():
    # Create a temporary data frame to hold importance related values
    feat_imp = dict(zip(X_train.columns, dict_best_estimators[this_best_sampler][target].feature_importances_))
    best_fi = list( zip( X_train.columns, dict_best_estimators[this_best_sampler][target].feature_importances_) ) 
    best_fi.sort( reverse=True, key=lambda x: x[1] )
    this_fi_name = target + "_feat_imp"
    df_temp = pd.DataFrame.from_dict(feat_imp, orient="index", columns=[this_fi_name])
    # With permutation importance, less biased for HIGH CARDINALITY DATA, BUT COMPUTATIONALLY EXPENSIVE. Uncomment the below lines if desired
    # Choose scoring as "roc_auc" or "f1"
    calc_perm_imp = permutation_importance( dict_best_estimators[this_best_sampler][target], X_train, y_train[target], scoring=scoring, n_repeats=these_n_repeats, random_state=this_rand_state, n_jobs=these_n_jobs)
    ###permutation_features = dict(zip(X_train.columns, calc_perm_imp.importances_mean))  # If doing this, then need to make a column for importances", "importances_mean", "importances_std"
    this_pi_mean_acc = target + "_acc_perm_imp_mean"
    this_pi_std_acc = target + "_acc_perm_imp_std"
    this_pi_med_acc = target + "_acc_perm_imp_median"
    this_pi_raw_acc = target + "_acc_perm_imp_raw"
    this_pi_mean_bal = target + "_bal_acc_perm_imp_mean"
    this_pi_std_bal = target + "_bal_acc_perm_imp_std"
    this_pi_med_bal = target + "_bal_perm_imp_median"
    this_pi_raw_bal = target + "_bal_acc_perm_imp_raw"
    this_pi_mean_f1 = target + "_f1_weighted_perm_imp_mean"
    this_pi_std_f1 = target + "_f1_weighted_perm_imp_std"
    this_pi_med_f1 = target + "_f1_weighted_perm_imp_median"
    this_pi_raw_f1 = target + "_f1_weighted_perm_imp_raw"
    this_pi_mean_mcc = target + "_mcc_perm_imp_mean"
    this_pi_std_mcc = target + "_mcc_perm_imp_std"
    this_pi_med_mcc = target + "_mcc_perm_imp_median"
    this_pi_raw_mcc = target + "_mcc_perm_imp_raw"
    this_pi_mean_roc = target + "_roc_perm_imp_mean"
    this_pi_std_roc = target + "_roc_perm_imp_std"
    this_pi_med_roc = target + "_roc_perm_imp_median"
    this_pi_raw_roc = target + "_roc_perm_imp_raw"
    print(calc_perm_imp)
    print(calc_perm_imp['accuracy'].importances)
    df_temp[this_pi_mean_acc] = calc_perm_imp['accuracy'].importances_mean
    df_temp[this_pi_std_acc] = calc_perm_imp['accuracy'].importances_std
    df_temp[this_pi_med_acc] = np.median(calc_perm_imp['accuracy'].importances, axis=1)
    df_temp[this_pi_raw_acc] = calc_perm_imp['accuracy'].importances.tolist()
    df_temp[this_pi_mean_bal] = calc_perm_imp['balanced_accuracy'].importances_mean
    df_temp[this_pi_std_bal] = calc_perm_imp['balanced_accuracy'].importances_std
    df_temp[this_pi_med_bal] = np.median(calc_perm_imp['balanced_accuracy'].importances, axis=1)
    df_temp[this_pi_raw_bal] = calc_perm_imp['balanced_accuracy'].importances.tolist()
    df_temp[this_pi_mean_f1] = calc_perm_imp['f1_weighted'].importances_mean
    df_temp[this_pi_std_f1] = calc_perm_imp['f1_weighted'].importances_std
    df_temp[this_pi_med_f1] = np.median(calc_perm_imp['f1_weighted'].importances, axis=1)
    df_temp[this_pi_raw_f1] = calc_perm_imp['f1_weighted'].importances.tolist()
    df_temp[this_pi_mean_mcc] = calc_perm_imp['mcc'].importances_mean
    df_temp[this_pi_std_mcc] = calc_perm_imp['mcc'].importances_std
    df_temp[this_pi_med_mcc] = np.median(calc_perm_imp['mcc'].importances, axis=1)
    df_temp[this_pi_raw_mcc] = calc_perm_imp['mcc'].importances.tolist()
    df_temp[this_pi_mean_roc] = calc_perm_imp['roc_auc'].importances_mean
    df_temp[this_pi_std_roc] = calc_perm_imp['roc_auc'].importances_std
    df_temp[this_pi_med_roc] = np.median(calc_perm_imp['roc_auc'].importances, axis=1)
    df_temp[this_pi_raw_roc] = calc_perm_imp['roc_auc'].importances.tolist()
    # Capture the median importances, and create a dictionary to sort a later dataframe by that median importance instead of mean (which was calculated in previous code versions) - db 20221130
    best_pi = list( zip( X_train.columns, np.median(calc_perm_imp[pick_method].importances, axis=1) ) )
    best_pi.sort( reverse=True, key=lambda x: x[1] )
    dict_target_top[target] = best_pi

    ###permutation_features = dict(zip(X_train.columns, calc_perm_imp))   # Captures a dictionary of dictionaries, internal keys are "importances", "importances_mean", "importances_std"
    
    ###permutation_features = dict(zip(X_train.columns, df_final_out['chi2_score'].to_numpy().tolist()))
    #this_col_name = target + '_perm_imp'
    #df_temp[this_col_name] = df_final_out['chi2_score'].to_numpy().tolist()
    ###df_pi = pd.DataFrame.from_dict(permutation_features, orient="index", columns=[this_col_name])
    ###df_final_out = df_final_out.join(df_pi, how='outer')
    df_final_out = df_final_out.join(df_temp, how='outer')
outname_df_final_out = fp_to_graphic_dir + "/" + this_out_name + "_picked_" + this_best_sampler + "_target_importance_values_per_features_" + pick_method + "_TRAIN.tsv"
df_final_out.to_csv( outname_df_final_out, sep='\t')
#df_final_out.to_feather

# Capture the cluster rankings from the TRAINING set
#print(dict_target_top)
df_ranked_clusters = pd.DataFrame()
df_ranked_clusters['rank'] = [ "rank_" + str(i+1) for i in range(0, len( list(dict_target_top.values())[0] )) ]
for key in dict_target_top.keys():
    these_genes = [ i[0] for i in dict_target_top[key] ]
    print("\n", key, dict_target_top[key])
    df_ranked_clusters[key] = dict_target_top[key]
    key_cluster = key + "_cluster"
    these_clusters = []
    key_contents = key + "_contents"
    these_contents = []
    for this_gene in these_genes:
        val1 = X_working.columns.get_loc(this_gene)
        these_clusters.append(val1)
        val2 = []
        for that_key in cluster_id_to_feature_ids.keys():
            if val1 in cluster_id_to_feature_ids[that_key]:
                val2 = cluster_id_to_feature_ids[that_key]
                #print( val1, that_key, cluster_id_to_feature_ids[that_key] )
        these_cols = X_working.columns[val2].tolist()
        these_contents.append(these_cols)
        print(this_gene, val1, len(these_cols), these_cols )
    df_ranked_clusters[key_cluster] = these_clusters
    df_ranked_clusters[key_contents] = these_contents
#ranks = [ "rank_" + str(i+1) for i in range(0,df_ranked_clusters.shape[0])]
#df_ranked_clusters['rank'] = ranks
#outname_df_ranked_clusters_out = fp_to_graphic_dir + "/" + this_out_name + "_picked_" + this_best_sampler + "_all_ranked_clusters_num_clust_" + str(this_cut) + "_" + pick_method + "_TRAINING.tsv"
outname_df_ranked_clusters_out = fp_to_graphic_dir + "/" + this_out_name + "_picked_" + this_best_sampler + "_all_ranked_clusters_num_clust_" + str(num_clusters) + "_" + pick_method + "_TRAINING.tsv"
df_ranked_clusters.to_csv( outname_df_ranked_clusters_out, sep='\t')

# FROM VALIDATION - Capture the chi2, Gini impurity based feature importances, and permutation importances of the features
df_final_out_2 = calculateCorrelationDataFrame( X_val, y_val, chi2, "chi2", 100 )
print('\nFeature Importances from Final Model by Permutation Importance on Validation Data')
dict_target_top_2 = {}
for target in y_val.columns.tolist():
    # Create a temporary data frame to hold importance related values
    feat_imp = dict(zip(X_val.columns, dict_best_estimators[this_best_sampler][target].feature_importances_))
    this_fi_name = target + "_feat_imp"
    df_temp = pd.DataFrame.from_dict(feat_imp, orient="index", columns=[this_fi_name])
    # With permutation importance, less biased for HIGH CARDINALITY DATA, BUT COMPUTATIONALLY EXPENSIVE. Uncomment the below lines if desired
    # Choose scoring as "roc_auc" or "f1"
    calc_perm_imp = permutation_importance( dict_best_estimators[this_best_sampler][target], X_val, y_val[target], scoring=scoring, n_repeats=these_n_repeats, random_state=this_rand_state, n_jobs=these_n_jobs)
    ###permutation_features = dict(zip(X_train.columns, calc_perm_imp.importances_mean))  # If doing this, then need to make a column for importances", "importances_mean", "importances_std"
    this_pi_mean_acc = target + "_acc_perm_imp_mean"
    this_pi_std_acc = target + "_acc_perm_imp_std"
    this_pi_med_acc = target + "_acc_perm_imp_median"
    this_pi_raw_acc = target + "_acc_perm_imp_raw"
    this_pi_mean_bal = target + "_bal_acc_perm_imp_mean"
    this_pi_std_bal = target + "_bal_acc_perm_imp_std"
    this_pi_med_bal = target + "_bal_perm_imp_median"
    this_pi_raw_bal = target + "_bal_acc_perm_imp_raw"
    this_pi_mean_f1 = target + "_f1_weighted_perm_imp_mean"
    this_pi_std_f1 = target + "_f1_weighted_perm_imp_std"
    this_pi_med_f1 = target + "_f1_weighted_perm_imp_median"
    this_pi_raw_f1 = target + "_f1_weighted_perm_imp_raw"
    this_pi_mean_mcc = target + "_mcc_perm_imp_mean"
    this_pi_std_mcc = target + "_mcc_perm_imp_std"
    this_pi_med_mcc = target + "_mcc_perm_imp_median"
    this_pi_raw_mcc = target + "_mcc_perm_imp_raw"
    this_pi_mean_roc = target + "_roc_perm_imp_mean"
    this_pi_std_roc = target + "_roc_perm_imp_std"
    this_pi_med_roc = target + "_roc_perm_imp_median"
    this_pi_raw_roc = target + "_roc_perm_imp_raw"
    #print(calc_perm_imp)
    df_temp[this_pi_mean_acc] = calc_perm_imp['accuracy'].importances_mean
    df_temp[this_pi_std_acc] = calc_perm_imp['accuracy'].importances_std
    df_temp[this_pi_med_acc] = np.median(calc_perm_imp['accuracy'].importances, axis=1)
    df_temp[this_pi_raw_acc] = calc_perm_imp['accuracy'].importances.tolist()
    df_temp[this_pi_mean_bal] = calc_perm_imp['balanced_accuracy'].importances_mean
    df_temp[this_pi_std_bal] = calc_perm_imp['balanced_accuracy'].importances_std
    df_temp[this_pi_med_bal] = np.median(calc_perm_imp['balanced_accuracy'].importances, axis=1)
    df_temp[this_pi_raw_bal] = calc_perm_imp['balanced_accuracy'].importances.tolist()
    df_temp[this_pi_mean_f1] = calc_perm_imp['f1_weighted'].importances_mean
    df_temp[this_pi_std_f1] = calc_perm_imp['f1_weighted'].importances_std
    df_temp[this_pi_med_f1] = np.median(calc_perm_imp['f1_weighted'].importances, axis=1)
    df_temp[this_pi_raw_f1] = calc_perm_imp['f1_weighted'].importances.tolist()
    df_temp[this_pi_mean_mcc] = calc_perm_imp['mcc'].importances_mean
    df_temp[this_pi_std_mcc] = calc_perm_imp['mcc'].importances_std
    df_temp[this_pi_med_mcc] = np.median(calc_perm_imp['mcc'].importances, axis=1)
    df_temp[this_pi_raw_mcc] = calc_perm_imp['mcc'].importances.tolist()
    df_temp[this_pi_mean_roc] = calc_perm_imp['roc_auc'].importances_mean
    df_temp[this_pi_std_roc] = calc_perm_imp['roc_auc'].importances_std
    df_temp[this_pi_med_roc] = np.median(calc_perm_imp['roc_auc'].importances, axis=1)
    df_temp[this_pi_raw_roc] = calc_perm_imp['roc_auc'].importances.tolist()
    
    ###permutation_features = dict(zip(X_train.columns, calc_perm_imp))   # Captures a dictionary of dictionaries, internal keys are "importances", "importances_mean", "importances_std"
    best_pi = list( zip( X_train.columns, np.median(calc_perm_imp[pick_method].importances, axis=1) ) )
    best_pi.sort( reverse=True, key=lambda x: x[1] )
    dict_target_top_2[target] = best_pi
    ###permutation_features = dict(zip(X_train.columns, df_final_out['chi2_score'].to_numpy().tolist()))
    #this_col_name = target + '_perm_imp'
    #df_temp[this_col_name] = df_final_out['chi2_score'].to_numpy().tolist()
    ###df_pi = pd.DataFrame.from_dict(permutation_features, orient="index", columns=[this_col_name])
    ###df_final_out = df_final_out.join(df_pi, how='outer')
    df_final_out_2 = df_final_out_2.join(df_temp, how='outer')
outname_df_final_out_2 = fp_to_graphic_dir + "/" + this_out_name + "_picked_" + this_best_sampler + "_target_importance_values_per_features_" + pick_method + "_VAL.tsv"
df_final_out_2.to_csv( outname_df_final_out_2, sep='\t')

# Capture the cluster rankings from the VALIDATION set
#print(dict_target_top_2)
df_ranked_clusters_2 = pd.DataFrame()
df_ranked_clusters_2['rank'] = [ "rank_" + str(i+1) for i in range(0, len( list(dict_target_top_2.values())[0] )) ]
for key in dict_target_top_2.keys():
    these_genes = [ i[0] for i in dict_target_top_2[key] ]
    print("\n", key, dict_target_top_2[key])
    df_ranked_clusters_2[key] = dict_target_top_2[key]
    key_cluster = key + "_cluster"
    these_clusters = []
    key_contents = key + "_contents"
    these_contents = []
    for this_gene in these_genes:
        val1 = X_working.columns.get_loc(this_gene)
        these_clusters.append(val1)
        val2 = []
        for that_key in cluster_id_to_feature_ids.keys():
            if val1 in cluster_id_to_feature_ids[that_key]:
                val2 = cluster_id_to_feature_ids[that_key]
                #print( val1, that_key, cluster_id_to_feature_ids[that_key] )
        these_cols = X_working.columns[val2].tolist()
        these_contents.append(these_cols)
        print(this_gene, val1, len(these_cols), these_cols )
    df_ranked_clusters_2[key_cluster] = these_clusters
    df_ranked_clusters_2[key_contents] = these_contents
#ranks = [ "rank_" + str(i+1) for i in range(0,df_ranked_clusters.shape[0])]
#df_ranked_clusters['rank'] = ranks
#outname_df_ranked_clusters_out_2 = fp_to_graphic_dir + "/" + this_out_name + "_picked_" + this_best_sampler + "_all_ranked_clusters_num_clust_" + str(this_cut) + "_" + pick_method + "_VAL.tsv"
outname_df_ranked_clusters_out_2 = fp_to_graphic_dir + "/" + this_out_name + "_picked_" + this_best_sampler + "_all_ranked_clusters_num_clust_" + str(num_clusters) + "_" + pick_method + "_VAL.tsv"
df_ranked_clusters_2.to_csv( outname_df_ranked_clusters_out_2, sep='\t')