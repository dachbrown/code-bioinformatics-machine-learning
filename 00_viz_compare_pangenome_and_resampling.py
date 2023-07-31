#!/usr/bin/env python3

# Data handling
from math import comb
from random import sample
import numpy as np
import pandas as pd
from sklearn import preprocessing
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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import BalancedRandomForestClassifier

#pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
"""
From ml_testing_imbalearn.py

This script reads a combined feature and label matrix to do machine learning. Uses the "ml_imba_test_clone" environment.


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
    "fp_ws_ec"      : '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_exclude_cloud.ftr',            # Filepath for with split, exclude cloud (15)
    "fp_ns_ec"      : '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_no_split_exclude_cloud.ftr',              # Filepath for no split, exclude cloud (15)
    "fp_ws_core"    : '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_core.ftr',                   # Filepath for with split, core
    "fp_ws_core_sc" : '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_core_and_soft_core.ftr',  # Filepath for with split, core and soft core
    "fp_ws_core_10" : '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_core_10.ftr',             # Filepath for with split, core to cloud 10
    "fp_ws_sh"      : '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_shell.ftr',                    # Filepath for with split, shell
    "fp_ws_sc_sh"   : '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_soft_core_and_shell.ftr',   # Filepath for with split, soft core and shell
    "fp_ws_total"   : '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_total.ftr',                 # Filepath for with split, total
    "fp_ws_shell_10": '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_shell_10.ftr'         # Filepath for with split, shell to cloud 10
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
# Choose the input filepaths to investigate
these_fps = dict_single
# Assign a filepath to the output directory
fp_to_graphic_dir = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/test_images"
# Assign number of estimators for the RandomForestClassifier
these_n_estimators = 100
# Assign other hyperparameters for the RandomForestClassifier
this_test_size = 0.2    # Float for percentage of samples as testing data set
this_rand_state = 1234  # For all random states
these_n_splits = 10      # For KFold
these_n_jobs = -1       # For all n_jobs, though 4 is also a good choice
num_chosen_genes = 10   # Integer for the number of chosen genes/features to include in plots and print outs
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
scoring = ['accuracy', 'balanced_accuracy', 'roc_auc']
# Select the primary scoring metric for choosing an estimator
choose_score = 'test_roc_auc'
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
# Either RandomForestClassifier is useful. As of 20221019, the default settings on both appear to be the same.
# From scikit-learn
#rfc = RandomForestClassifier(n_estimators=these_n_estimators, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
# From imblearn
rfc = BalancedRandomForestClassifier(n_estimators=these_n_estimators, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
####################
####################


####################
### EXECUTION
####################
# Instantiate output dataframe for metrics
df_out_full_estimator = pd.DataFrame()

for designator in these_fps.keys():
    this_fp = these_fps[designator]
    # Assign a name that will receive prefixes and suffixes on all outfiles
    this_out_name = designator + "_ne_" + str(these_n_estimators)
    # Report user inputs
    print(
        "\nName", designator,
        "\nFilepath", this_fp,
        "\nLabel Column", this_col,
        "\nMultilabel Columns", these_multilabels,
        "\nTest Size", this_test_size,
        "\nEstimators", these_n_estimators,
        "\nKFold Splits", these_n_splits,
        "\nNum. Top Genes (features)", num_chosen_genes,
        "\n"
    )

    # Create features and labels
    X_data, y_data, unsplit_data = splitFeaturesLabels(this_fp, "asm_level", -400)
    showColStats(y_data, this_col)

    # Copy feature and label dataframes for safety, subsetting all labels to interested labels here.
    X_working = X_data.copy()
    y_working = y_data[these_multilabels].copy()

    # Determine method for splitting into training and testing data sets
    msss = MultilabelStratifiedShuffleSplit( n_splits=1, test_size=this_test_size, random_state=this_rand_state )

    # Explicitly create empty dataframes for safety
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    # Split & stratify into training and testing to prevent data leakage and preserve real-world label ratios
    for train_index, test_index in msss.split(X_working, y_working):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_working.iloc[train_index], X_working.iloc[test_index]
        y_train, y_test = y_working.iloc[train_index], y_working.iloc[test_index]

    # Determine sampling methods
    us_rnd = RandomUnderSampler( sampling_strategy='not minority', random_state=this_rand_state )
    os_rnd = RandomOverSampler( sampling_strategy='not majority', random_state=this_rand_state )
    os_smote = SMOTE ( random_state=this_rand_state )
    combo_smoteenn = SMOTEENN( random_state=this_rand_state )
    combo_smotetmk = SMOTETomek( random_state=this_rand_state )
    #sample_methods_list = [ us_rnd, os_rnd, os_smote, combo_smoteenn, combo_smotetmk ]
    sample_methods_list = [ us_rnd, os_rnd, combo_smoteenn ]

    # Note that imblearn does not allow 'multi-label' indicator data, so we are getting around that by treating each target column individually
    mskf = MultilabelStratifiedKFold ( n_splits=these_n_splits, shuffle=True, random_state=this_rand_state )
    these_mskf_splits = list(mskf.split(X_train, y_train))    # This is required here, as a multilabel-indicator is required. The splits are passed to a later cv= within cross_validate
    sample_methods_list = [ us_rnd, os_rnd, combo_smoteenn ]

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
    #dict_sampler = {}
    #dict_sampler['y_test'] = y_test
    # Run pipeline and collect metrics
    for sampler in sample_methods_list:
        model = make_pipeline(sampler, rfc)
        dict_targets_cv = {}
        all_y_pred = [] # Just a placeholder, as empty nparrays have constraints
        all_y_score = [] # Just a placeholder, as empty nparrays have constraints
        # Perform cross validation while fitting the estimators, results are stored by target
        for target in y_train.columns.tolist():
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
            fold_median = ( np.abs( cv_return[choose_score] - np.median(cv_return[choose_score]) ) ).argmin()
            print('\nFor', target)
            print( "\nThe median value for ", choose_score, " is ", str(np.median(cv_return[choose_score])), " and the closest estimator score to that median value is ", str(cv_return[choose_score][fold_median]), " from fold ", str(fold_median), "\n" )
            choose_fold = fold_median
            #print(cv_return)
            for score in cv_return.keys():
                if score != 'estimator':
                    this_mean = np.mean(cv_return[score], axis=0)
                    this_std = np.std(cv_return[score])
                    print( score, "mean: ", this_mean, " +/- ", this_std )
                else:
                    print("The following OOB should be similar to the mean scores")
                    print("OOB", cv_return['estimator'][choose_fold][-1].oob_score_)
            # Assign to dictionary
            dict_targets_cv[target]['best_estimator'] = cv_return['estimator'][choose_fold][-1] # Has already been fitted.
            dict_targets_cv[target]['best_predictions'] = cv_return['estimator'][choose_fold][-1].predict(X_test)
            dict_targets_cv[target]['best_scores'] = cv_return['estimator'][choose_fold][-1].predict_proba(X_test)
            # Report the classification report from imblearn
            print('\nimblearn classification report for', sampler)
            print('\nFor', target)
            print(classification_report_imbalanced(y_test[target], dict_targets_cv[target]['best_predictions'], zero_division=0))
            # Access the pipeline object for the estimator to report feature importances
            # cross_validate returns a dictionary, requiring a the 'estimator' key to access all the returned, fitted estimators. Each estimator is accessed via integer for the CV fold step. When a pipeline is used, the desired estimator is found via [integer][-1][1] to represent the final step of the pipeline and the estimator
            #print(sorted(cv_return['estimator'][choose_fold][-1].feature_importances_, reverse=True, key=lambda x: x[1])[:num_chosen_genes])
            #print('see fold genes above for fold ', choose_fold)
            print("****")
        # Stack the predictions and scores from the best fold classifier on a per target basis PSEUDOCODE np.hstack(fold0 target0, fold2 target1, etc.)
        # Process data according to https://scikit-learn.org/stable/modules/model_evaluation.html#multi-label-case
        for key in dict_targets_cv.keys():
            all_y_score.append(dict_targets_cv[key]['best_scores'])
            if list(dict_targets_cv.keys()).index(key) == 0:
                all_y_pred = np.transpose([dict_targets_cv[key]['best_predictions']])
            else:
                all_y_pred = np.hstack( (all_y_pred, np.transpose([dict_targets_cv[key]['best_predictions']])) )
        all_y_score = np.transpose([y_pred[:, 1] for y_pred in all_y_score])

        #[str(sampler).split('(')[0]]['all_y_score'] = all_y_score
        #dict_sampler[str(sampler).split('(')[0]]['all_y_preds'] = all_y_pred

        # Visuals
        # Please reference https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
        macro_auc_roc = roc_auc_score(y_test, all_y_score, average='macro' )
        
        fig, axes = plt.subplots(1, len(these_multilabels), figsize=(15, 5))
        fig.tight_layout()
        axes = axes.ravel()
        for i in range(len(these_multilabels)):
            this_sub = ConfusionMatrixDisplay.from_predictions(
                y_test.iloc[:,i],
                all_y_pred[:,i],
                display_labels=[0,1],
                normalize='true'
            )
            this_sub.plot(ax=axes[i])#, values_format='.4g')
            this_sub.ax_.set_title(y_test.columns.tolist()[i])
            this_sub.im_.colorbar.remove()
            plt.close()
        fig.subplots_adjust(wspace=0.4, left=0.1, top=0.9, bottom=0.1)
        #fig.suptitle( str(sampler) + " MultiLabel " + this_fp )
        fig.suptitle( str(sampler).split('(')[0] + "- BRFC - MultiLabel - Macro AUROC " + str(round(macro_auc_roc, 3)) + " " + this_fp )
        fig.colorbar(this_sub.im_, ax=axes)
        out_fp = fp_to_graphic_dir + "/" + str(sampler).split('(')[0] + "_macro_auroc_" + str(round(macro_auc_roc, 3)) + "_" + this_out_name + ".png"
        plt.savefig( out_fp, format='png', dpi='figure', pad_inches=0.1 )
        #plt.show()
        plt.close()

        print(sampler, " MultiLabel Metrics")
        # NOTE "multilabel_confusion_matrix" from scikit-learn is very very broken, potentially due to numpy/numba/Python conflicts - 20221019 db
        print('cr')
        print( classification_report(y_test, all_y_pred, zero_division=0, target_names=y_test.columns.tolist()) )
        
        dict_scores = {
            'accuracy_score'                            :   accuracy_score(y_test, all_y_pred),
            'f1_score_avg_weighted'                     :   f1_score(y_test, all_y_pred, average='weighted'),
            'f1_score_avg_macro'                        :   f1_score(y_test, all_y_pred, average='macro'),
            'fbeta_score_betadot5_avg_weighted'         :   fbeta_score(y_test, all_y_pred, beta=0.5, average='weighted'),
            'fbeta_score_betadot5_avg_macro'            :   fbeta_score(y_test, all_y_pred, beta=0.5, average='macro'),
            'hamming_loss'                              :   hamming_loss(y_test, all_y_pred),
            'jaccard_score_weighted'                    :   jaccard_score(y_test, all_y_pred, average='weighted'),
            'jaccard_score_macro'                       :   jaccard_score(y_test, all_y_pred, average='macro'),
            'log_loss'                                  :   log_loss(y_test, all_y_pred),
            'precision_recall_fscore_support_none'      :   precision_recall_fscore_support(y_test, all_y_pred),
            'precision_recall_fscore_support_weighted'  :   precision_recall_fscore_support(y_test, all_y_pred, average='weighted' ),
            'precision_recall_fscore_support_macro'     :   precision_recall_fscore_support(y_test, all_y_pred, average='macro' ),
            'roc_auc_weighted'                          :   roc_auc_score(y_test, all_y_score, average='weighted' ),
            'roc_auc_macro'                             :   roc_auc_score(y_test, all_y_score, average='macro' ),
            'zero_one_loss'                             :   zero_one_loss(y_test, all_y_pred, normalize=True),
            'average_precision_score'                   :   average_precision_score(y_test, all_y_score),
            'sparsity'                                  :   float(X_train.astype(pd.SparseDtype("int", 0)).sparse.density)
        }

        for score in dict_scores.keys():
            print("\n", score)
            print(dict_scores[score])
        
        col_name = designator + "_" + str(sampler).split('(')[0]
        df_out_full_estimator[col_name] = [ dict_scores[score] for score in dict_scores.keys() ]
        df_out_full_estimator['scores'] = list(dict_scores.keys())
        df_out_full_estimator.set_index( 'scores', inplace=True )
        
# Save and output metrics
df_out_full_estimator.reset_index(inplace=True)
outname_df_out_full_estimator = fp_to_graphic_dir + "/" + designator + "_df_all_sampler_subset_scores_combined_estimator.tsv"
df_out_full_estimator.to_csv(outname_df_out_full_estimator, sep='\t')
    #[str(sampler).split('(')[0]]['targets_dict'] = dict_targets_cv

    #print('\nFeature Importances from Final Model by Permutation Importance on Test Data')
    #for target in dict_targets_cv.keys():
        ## With permutation importance, less biased for HIGH CARDINALITY DATA, BUT COMPUTATIONALLY EXPENSIVE. Uncomment the below lines if desired
        #calc_perm_imp = permutation_importance( dict_targets_cv[target]['best_estimator'], X_test, y_test[target], n_repeats=10, random_state=this_rand_state, n_jobs=these_n_jobs)
        #permutation_features = list(zip(X_train.columns, calc_perm_imp.importances_mean))
        #permutation_features.sort(reverse=True, key=lambda x: x[1])
        #best_permutation = [ pair[0] for pair in permutation_features ]    # Will be the same length as best_select, which is determined by the user indicated percentile variable "this_perc_best"
        #dict_best_permutation = { i[0] : i[1] for i in best_permutation }
        #df_best_permutation = pd.DataFrame.from_dict(dict_best_permutation, orient="index", columns=['permutation_importance'])
        #names_best_permutation = list(dict_best_permutation.keys())
        #print("fold", fold, permutation_features[:num_chosen_genes])
        #print("fold", fold, best_permutation[:num_chosen_genes])