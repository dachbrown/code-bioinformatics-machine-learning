#!/usr/bin/env python3

import numpy as np
import pandas as pd
import math
import feather
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from matplotlib import pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.utils.multiclass import type_of_target
from sklearn.multioutput import MultiOutputClassifier
from sklearn import preprocessing

"""
From ml_test_random_forest_20221012.py

This script reads a combined feature and label matrix to do machine learning. Uses the "galick_gun" environment.


### TNT - use label binarizer and onevsrestclassifier, then get scores???

# Clean this up here into functions
### Build
# StratifiedKFold for sampling
# Upsampler
# Downsampler
# compare the above 3 with the previous for 4 total via
# current accuracy score (old)
# confusion matrix
# auroc visualizer
# Cohens kappa?
# Save the results in a better output format
# Build this into truly multiclass
# Build a script that takes the outputs and compares against CARD/UNIPROT/PHASTER? for verification of pathways?
# make sure when saving importance to also collect the value, not just the name

### BUGS
# Some weird behavior with my implementation of percentile and the sklearn percentile. have tried both round and math.ceil, but the values can be slightly different. is the answer math.floor?
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

### NOTE samplers are used after the train test split, so i should run against stratified split
# Upsamples the data to create balanced classes
#def dataUpsampler

# Downsamples the data to create balanced classes
#def dataDownsampler

# Implementation of sklearn StratifiedKFold
#def dataStratifiedSampler

# Make an RF classifier while printing and returning useful values
def makeRFC( untrained_parameterized_rfc, up_rfc_name, these_train_features, these_test_features, these_train_labels, these_test_labels ):
    """
    Takes an initialized RandomForestClassifier and trains it on provided data. Returns the classifier and predictions.
    
    untrained_parameterized_rfc The untrained, parameterized RandomForestClassifer (object)
    up_rfc_name                 The reference name for the trained classifier (string)
    these_train_features        The training features (matrix)
    these_test_features         The testing features (matrix)
    these_train_labels          The training labels (matrix)
    these_test_labels           The testing labels (matrix)
    """
    # Train the classifier
    trained_rfc = untrained_parameterized_rfc.fit(these_train_features, these_train_labels)

    # Check the classifier score
    rfc_y_pred = trained_rfc.predict(these_test_features)
    rfc_scores = cross_val_score(trained_rfc, these_test_features, these_test_labels, cv=5)
    print("Calculated metrics for", up_rfc_name)
    print("The below values should be similar for Random Forest")
    print("CrossVal", rfc_scores.mean())
    print("OOB", trained_rfc.oob_score_)
    print("Classification Report for", up_rfc_name)
    print(classification_report(these_test_labels, rfc_y_pred))

    # Return the trained classifier and the predicted labels
    return( trained_rfc, rfc_y_pred )

#def performMultiClass

#def importanceImpurityBased

#def importancePermutation

# Report mean and standard deviation for the scores found in a "cross_validate' dictionary.
def statsCrossValidate( cross_validate_dict, name_string_clf, choose_metric ):
    """
    Reports the mean and standard deviation for scores found in a "cross_validate" dictionary
    Returns the 

    Input:
    - cross_validate_dict   The dictionary returned by a "cross_validate" function call
    - name_string_clf       A string for the name of the classifier that was cross-validated
    - choose_metric         A string for the name of the chosen test metric to find the median
    """
    print( "Results of cross-validation for ", name_string_clf )
    for score in cross_validate_dict.keys():
        if score != "estimator":
            this_mean = np.mean(cross_validate_dict[score], axis=0)
            this_std = np.std(cross_validate_dict[score])
            print( score, "mean: ", this_mean, " +/- ", this_std )
        else:
            this_index = ( np.abs( cross_validate_dict[choose_metric] - np.median(cross_validate_dict[choose_metric]) ) ).argmin() # Identifies the index of the score closest to the median of all the scores.
            print( "The index of the median value ", str(np.median(cross_validate_dict[choose_metric])), " for '", choose_metric, "' is ", str(this_index) )
    return(cross_validate_dict['estimator'][this_index])

#def computeAccuracyScore

#def computeConfusionMatrix

#def computeAUROCvisualizer

# Compares a cross-validated classifier against itself via AUROC and confusion matrix. Please reference https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
def visualizeCrossValidatedClassifier( this_classifier, this_classifier_warm, this_cv, X_features, y_labels, this_roc_title, this_roc_outfile, this_cm_title, this_cm_outfile ):
    """
    Modified from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

    Returns a dataframe of the important features, their impurity-based importance, and the mean impurity-based importance per feature from all cross validation folds
    """

    # Generates an AUROC visual (from the classifier estimator) comparing the results of the cross-validation splits
    y_test_all = []
    y_pred_all = []

    df_all_fold_feat_imp = pd.DataFrame()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    #plt.figure(figsize=(8,6), dpi=100)
    for i, (train, test) in enumerate(this_cv.split(X_features, y_labels)):
        print(i)
        #this_classifier.fit(X_features[train], y_labels[train]) # Assumes a np.array
        this_classifier.fit(X_features.iloc[train], y_labels.iloc[train]) # Assumes pandas dataframe
        #this_classifier_warm.fit(X_features.iloc[train], y_labels.iloc[train]) # Assumes pandas dataframe
        viz = RocCurveDisplay.from_estimator(
            this_classifier,
            #X_features[test],  # Assumes a np.array
            X_features.iloc[test],  # Assumes a pandas dataframe
            #y_labels[test],    # Assumes a np.array
            y_labels.iloc[test],    # Assumes a pandas dataframe
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        #print(viz.roc_auc)
        #these_y_pred = this_classifier.predict(X_features[test])   # Assumes a np.array
        these_y_pred = this_classifier.predict(X_features.iloc[test])   # Assumes a pandas dataframe
        y_pred_all.append(these_y_pred)
        #y_test_all.append(y_labels[test])   # Assumes a np.array
        y_test_all.append(y_labels.iloc[test])  # Assumes a pandas dataframe
        #print("y_pred_test_done")

        # Capture the feature importances
        #this_fold_feat_imp = list(zip(X_features.columns, this_classifier.feature_importances_))   # Assumes a pandas dataframe
        #this_fold_feat_imp.sort(reverse=True, key=lambda x: x[1])
        #dict_this_fold_feat_imp = { X_features.columns.tolist()[i] : this_classifier.feature_importances_.tolist()[i] for i in range(len(X_features.columns.tolist())) }
        dict_this_fold_feat_imp = dict(zip(X_features.columns, this_classifier.feature_importances_))
        col_name_string = 'fold_' + str(i) + '_impurity_importance'
        #df_this_fold_feat_imp = pd.DataFrame.from_dict(dict_this_fold_feat_imp, orient="index", columns=[col_name_string])
        add_estimators = len(this_classifier.estimators_)  # Capture number of estimators for the unupdated classifier
        if i==0:
            #df_all_fold_feat_imp = df_this_fold_feat_imp
            df_all_fold_feat_imp = pd.DataFrame.from_dict(dict_this_fold_feat_imp, orient="index", columns=[col_name_string])
            this_classifier_warm.fit(X_features.iloc[train], y_labels.iloc[train]) # Assumes pandas dataframe
            #print(len(this_classifier_warm.estimators_))
            #print('see me once', i)
        else:
            #df_all_fold_feat_imp = df_all_fold_feat_imp.join(df_this_fold_feat_imp, how='outer')
            df_all_fold_feat_imp[col_name_string] = [ dict_this_fold_feat_imp[key] for key in dict_this_fold_feat_imp.keys() ]
            more_estimators = len(this_classifier_warm.estimators_)
            #print(more_estimators)
            more_estimators += add_estimators
            #print(more_estimators)
            this_classifier_warm.set_params(n_estimators=more_estimators)
            #print(len(this_classifier_warm.estimators_))
            this_classifier_warm.fit(X_features.iloc[train], y_labels.iloc[train]) # Assumes pandas dataframe
            #print(len(this_classifier_warm.estimators_))
            #print('see me several', i)

        #print("end ", i)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=this_roc_title,
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig( this_roc_outfile, format='png', dpi='figure', pad_inches=0.1 )
    plt.show()

    
    # Generates a Confusion Matrix visual from ALL the classifier predictions during cross validation
    #plt.figure(figsize=(8,6), dpi=100)
    y_pred_all = tuple( i for i in y_pred_all )
    y_pred_all = np.concatenate(y_pred_all, axis=None)
    y_test_all = tuple( i for i in y_test_all )
    y_test_all = np.concatenate(y_test_all, axis=None)
    #fig2, ax2 = plt.subplots()
    cm_disp_predictions = ConfusionMatrixDisplay.from_predictions(
        y_test_all,
        y_pred_all,
        display_labels=this_classifier.classes_,
        normalize='true',    # Normalize over rows for % of true values
        #ax=ax2
    )
    ax2=plt.gca()
    ax2.set_title(this_cm_title)
    plt.tight_layout()
    #cm_disp_predictions.plot()
    plt.savefig( this_cm_outfile, format='png', dpi='figure', pad_inches=0.1 )
    plt.show()

    # Prints a classification report and other useful information
    these_scores = cross_val_score(this_classifier, X_features, y_labels, cv=this_cv.split(X_features, y_labels)) # Similar syntax as above
    print("The below should be similar for Random Forest")
    print("CrossVal", these_scores.mean())
    print("OOB", this_classifier.oob_score_)
    print("Classification Report Associated with ", this_cm_title)
    print(classification_report(y_test_all, y_pred_all))

    # Returns the estimator?? based on functionality of 0.2
    ### TNT - get the importance here as well???

    # Returns the important features from each classifier run
    df_all_fold_feat_imp['mean_feature_importance_impurity'] = df_all_fold_feat_imp.mean(axis=1)
    return(df_all_fold_feat_imp, this_classifier_warm)


# Compares a cross-validated classifier against itself via AUROC and confusion matrix. Please reference https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
def visualizeCrossValidatedClassifier2( cross_validate_dict, this_cv, X_features, y_labels, this_roc_title, this_roc_outfile, this_cm_title, this_cm_outfile ):
    """
    Modified from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

    Returns a dataframe of the important features, their impurity-based importance, and the mean impurity-based importance per feature from all cross validation folds
    """

    # Generates an AUROC visual (from the classifier estimator) comparing the results of the cross-validation splits
    y_test_all = []
    y_pred_all = []

    df_all_fold_feat_imp = pd.DataFrame()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    #plt.figure(figsize=(8,6), dpi=100)
    for i, (train, test) in enumerate(this_cv.split(X_features, y_labels)):
    #for this_classifier in cross_validate_dict['estimator']:
        print(i)
        this_classifier = cross_validate_dict['estimator'][i]
        #this_classifier.fit(X_features[train], y_labels[train]) # Assumes a np.array
        #this_classifier.fit(X_features.iloc[train], y_labels.iloc[train]) # Assumes pandas dataframe
        #this_classifier_warm.fit(X_features.iloc[train], y_labels.iloc[train]) # Assumes pandas dataframe
        """
        # For estimator
        viz = RocCurveDisplay.from_estimator(
            this_classifier,
            #X_features[test],  # Assumes a np.array
            X_features.iloc[test],  # Assumes a pandas dataframe
            #y_labels[test],    # Assumes a np.array
            y_labels.iloc[test],    # Assumes a pandas dataframe
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        """
        #these_y_pred = this_classifier.predict(X_features[test])   # Assumes a np.array
        these_y_pred = this_classifier.predict(X_features.iloc[test])   # Assumes a pandas dataframe
        viz = RocCurveDisplay.from_predictions(
            #this_classifier,
            #X_features[test],  # Assumes a np.array
            #X_features.iloc[test],  # Assumes a pandas dataframe
            #y_labels[test],    # Assumes a np.array
            y_labels.iloc[test],    # Assumes a pandas dataframe
            these_y_pred,
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        #print(viz.roc_auc)
        #these_y_pred = this_classifier.predict(X_features[test])   # Assumes a np.array
        #these_y_pred = this_classifier.predict(X_features.iloc[test])   # Assumes a pandas dataframe
        y_pred_all.append(these_y_pred)
        #y_test_all.append(y_labels[test])   # Assumes a np.array
        y_test_all.append(y_labels.iloc[test])  # Assumes a pandas dataframe
        #print("y_pred_test_done")

        # Capture the feature importances
        #this_fold_feat_imp = list(zip(X_features.columns, this_classifier.feature_importances_))   # Assumes a pandas dataframe
        #this_fold_feat_imp.sort(reverse=True, key=lambda x: x[1])
        #dict_this_fold_feat_imp = { X_features.columns.tolist()[i] : this_classifier.feature_importances_.tolist()[i] for i in range(len(X_features.columns.tolist())) }
        dict_this_fold_feat_imp = dict(zip(X_features.columns, this_classifier.feature_importances_))
        col_name_string = 'fold_' + str(i) + '_impurity_importance'
        #df_this_fold_feat_imp = pd.DataFrame.from_dict(dict_this_fold_feat_imp, orient="index", columns=[col_name_string])
        #add_estimators = len(this_classifier.estimators_)  # Capture number of estimators for the unupdated classifier
        if i==0:
            #df_all_fold_feat_imp = df_this_fold_feat_imp
            df_all_fold_feat_imp = pd.DataFrame.from_dict(dict_this_fold_feat_imp, orient="index", columns=[col_name_string])
            #this_classifier_warm.fit(X_features.iloc[train], y_labels.iloc[train]) # Assumes pandas dataframe
            #print(len(this_classifier_warm.estimators_))
            #print('see me once', i)
        else:
            #df_all_fold_feat_imp = df_all_fold_feat_imp.join(df_this_fold_feat_imp, how='outer')
            df_all_fold_feat_imp[col_name_string] = [ dict_this_fold_feat_imp[key] for key in dict_this_fold_feat_imp.keys() ]
            #more_estimators = len(this_classifier_warm.estimators_)
            #print(more_estimators)
            #more_estimators += add_estimators
            #print(more_estimators)
            #this_classifier_warm.set_params(n_estimators=more_estimators)
            #print(len(this_classifier_warm.estimators_))
            #this_classifier_warm.fit(X_features.iloc[train], y_labels.iloc[train]) # Assumes pandas dataframe
            #print(len(this_classifier_warm.estimators_))
            #print('see me several', i)

        #print("end ", i)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=this_roc_title,
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig( this_roc_outfile, format='png', dpi='figure', pad_inches=0.1 )
    plt.show()

    
    # Generates a Confusion Matrix visual from ALL the classifier predictions during cross validation
    #plt.figure(figsize=(8,6), dpi=100)
    y_pred_all = tuple( i for i in y_pred_all )
    y_pred_all = np.concatenate(y_pred_all, axis=None)
    y_test_all = tuple( i for i in y_test_all )
    y_test_all = np.concatenate(y_test_all, axis=None)
    #fig2, ax2 = plt.subplots()
    cm_disp_predictions = ConfusionMatrixDisplay.from_predictions(
        y_test_all,
        y_pred_all,
        display_labels=this_classifier.classes_,
        normalize='true',    # Normalize over rows for % of true values
        #ax=ax2
    )
    ax2=plt.gca()
    ax2.set_title(this_cm_title)
    plt.tight_layout()
    #cm_disp_predictions.plot()
    plt.savefig( this_cm_outfile, format='png', dpi='figure', pad_inches=0.1 )
    plt.show()

    # Prints a classification report and other useful information
    these_scores = cross_val_score(this_classifier, X_features, y_labels, cv=this_cv.split(X_features, y_labels)) # Similar syntax as above
    print("The below should be similar for Random Forest")
    print("CrossVal", these_scores.mean())
    print("OOB", this_classifier.oob_score_)
    print("Classification Report Associated with ", this_cm_title)
    print(classification_report(y_test_all, y_pred_all))

    # Returns the estimator?? based on functionality of 0.2
    ### TNT - get the importance here as well???

    # Returns the important features from each classifier run
    df_all_fold_feat_imp['mean_feature_importance_impurity'] = df_all_fold_feat_imp.mean(axis=1)
    return(df_all_fold_feat_imp)

# Calculate the correlation for the dataset
def calculateCorrelationDataFrame( X_train, y_train, score_func_obj, score_func_name_str, percentile ):
    this_select = SelectPercentile(score_func=score_func_obj, percentile=percentile)  #this_select = SelectPercentile(score_func=mutual_info_classif, percentile=this_perc_best)
    this_select.fit(X_train, y_train)
    #X_train_best = this_select.transform(X_train)
    #X_test_best = this_select.transform(X_test)
    #names_best_select = this_select.get_feature_names_out().tolist()
    #corr_feature_scores = list(zip(X_train.columns, this_select.scores_))
    #select_features.sort(reverse=True, key=lambda x: x[1])
    #best_select = [ i for i in select_features[:len(names_best_select)] ]    # Will be the same length as best_select, which is determined by the user indicated percentile variable "this_perc_best"
    #dict_this_select = { X_train.columns.tolist()[i] : this_select.scores_.tolist()[i] for i in range(len(X_train.columns.tolist())) }
    dict_this_select = dict(zip(X_train.columns, this_select.scores_))
    col_name_str = score_func_name_str + "_score"
    df_this_select = pd.DataFrame.from_dict(dict_this_select, orient="index", columns=[col_name_str])

    return(df_this_select)

# Compares 2 classifiers (built on the same dataset labels, so a head to head prediction) via AUROC on the same data
def compareAUROC( out_file_name, clf_1, clf_1_name, clf_1_features, clf_1_pred_labels, clf_2, clf_2_name, clf_2_features, clf_2_pred_labels, test_labels ):
    """
    Compares 2 classifiers built on the same dataset labels via a combined AUROC display plot
    
    out_file_name       The output file name for the saved image (string)
    clf_1               The first classifier (object)
    clf_1_name          The first classifier's name (string)
    clf_1_features      The input features for training the first classifier (matrix)
    clf_1_pred_labels   The predicted labels from the first classifier (matrix)
    clf_2               The second classifier (object)
    clf_2_name          The second classifier's name (string)
    clf_2_features      The input features for training the second classifier (matrix)
    clf_2_pred_labels   The predicted labels from the second classifer (matrix)
    test_labels         The "true" labels from the reserved testing dataset
    """
    plt.figure(figsize=(8,6), dpi=100)
    ax = plt.gca() # Get current axes
    clf_1_name_estimator = clf_1_name + " Estimator"
    roc_plot_estimator_1 = RocCurveDisplay.from_estimator(
        clf_1,
        clf_1_features,
        test_labels,
        name=clf_1_name_estimator,
        ax=ax
    )
    ax = plt.gca() # Get current axes
    
    clf_1_name_predictions = clf_1_name + " Predictions"
    roc_plot_predictions_1 = RocCurveDisplay.from_predictions(
        test_labels,
        clf_1_pred_labels,
        name=clf_1_name_predictions,
        ax=ax
    )
    ax = plt.gca() # Get current axes

    clf_2_name_estimator = clf_2_name + " Estimator"
    roc_plot_estimator_2 = RocCurveDisplay.from_estimator(
        clf_2,
        clf_2_features,
        test_labels,
        name=clf_2_name_estimator,
        ax=ax
    )
    ax = plt.gca() # Get current axes

    clf_2_name_predictions = clf_2_name + " Predictions"
    roc_plot_predictions_2 = RocCurveDisplay.from_predictions(
        test_labels,
        clf_2_pred_labels,
        name=clf_2_name_predictions,
        ax=ax
    )
    title = clf_1_name + " vs " + clf_2_name
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig( out_file_name, format='png', dpi='figure', pad_inches=0.1)
    plt.show()

# Compares lists of lists through sets to arrive at union, intersection, and difference. Accepts a list of lists
def checkSets(input_list):
    input_list = [ set(i) for i in input_list ]
    try:
        # Capture the intersections
        intersections   =   set.intersection(*input_list)

        differences = []
        for minor_set in input_list:
            differences.extend( list(minor_set - intersections) )
        # Format outputs
        intersections   =   sorted(list(intersections))
        differences     =   sorted(list(set(differences)))
        total_union     =   sorted(list(set.union(*input_list)))
        return(intersections, differences, total_union)
        #return(intersections, differences)
    except:
        return("Input list variable must be filled and iterable.")

# Print the ".describe()" and ".value_counts()" from pandas.
def showColStats(dataframe, col_name):
    """
    Shows useful column stats. No variable output.
    """
    print("\n", col_name)
    print(dataframe[col_name].describe())
    print(dataframe[col_name].value_counts())
####################

####################
### INPUTS
####################
# User determined values for specific parameters
"""
this_label = [
    'aminoglycosides',
    #'beta_lactam_combination_agents',
    #'cephems',                         # Dropped as only 1 instance. See "ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2."
    'folate_pathway_antagonists',
    #'macrolides',
    #'nucleosides',
    #'penicillins',                     # Dropped as only 1 instance. See "ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2."
    #'phenicols',
    #'quinolones',
    'tetracyclines',
    'other',
    #'no_resistance'
]
"""
#this_label = 'MDR_bin'
this_label = "MDR_classes_drop_bla"
#this_label = 'no_resistance' # beta lactam bad choice, everything has it, maybe ami or fol? trying tet
#this_label = 'mcr_family'
#this_label = 'amiblafol_bin'
this_test_size = 0.5    # Float for percentage of samples as testing data set
this_k_best = 20        # Integer for flat number of best features
this_perc_best = 20   # NOT a float, an INT for percentile of retained best features
this_rand_state = 1234  # For all random states
these_n_splits = 4      # For KFold
these_n_jobs = -1       # For all n_jobs, though 4 is also a good choice
these_n_estimators = 50 # For all n_estimators
num_chosen_genes = 50   # Integer for the number of chosen genes/features to include in plots and print outs
fp_to_graphic_dir = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/final_images"
keep_threshold = 8  # Integer for the minimum class prevalence to include in the testing
####################

####################
### FILEPATHS TO DATASETS
# Filepath for with split, exclude cloud (15)
fp_ws_ec = '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_exclude_cloud.ftr'
# Filepath for no split, exclude cloud (15)
fp_ns_ec = '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_no_split_exclude_cloud.ftr'
# Filepath for with split, core
fp_ws_core = '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_core.ftr'
# Filepath for with split, core and soft core
fp_ws_core_sc = '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_core_and_soft_core.ftr'
# Filepath for with split, core to cloud 10
fp_ws_core_10 = '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_core_10.ftr'
# Filepath for with split, shell
fp_ws_sh = '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_shell.ftr'
# Filepath for with split, soft core and shell
fp_ws_sc_sh = '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_soft_core_and_shell.ftr'
# Filepath for with split, total
fp_ws_total = '/Users/dbrow208/Documents/galick_gun/test_prokka_roary/final_scripts/subset_900_with_split_total.ftr'
####################

####################
### EXECUTION
####################
# Instantiate the feature and label matrices
#print("\nFilepath", fp_ws_ec, "\nLabel", this_label, "\nPercentile Kept", this_perc_best, "\nTest Size", this_test_size, "\n")
#features_ws_ec, labels_ws_ec, data_ws_ec = splitFeaturesLabels(fp_ws_ec, "asm_level", -400)
#print("The below is a sanity check, comparing labels and original data frame, respectively.")
#showColStats(labels_ws_ec, this_label)
#showColStats(data, this_label)
#print("\n\nGeneral stats for 'MDR_bin'\n")
#showColStats(labels_ws_ec, "MDR_bin")
#print("\n\nGeneral stats for 'amiblafol_bin'\n")
#showColStats(labels_ws_ec, "amiblafol_bin")
#print("\n\nGeneral stats for 'resistance_classes'\n")
#showColStats(labels_ws_ec, "resistance_classes")

"""
# For multilabel, deprecated
print("\n\nGeneral stats for ", str(this_label), "\n")
for i in this_label:
    showColStats(labels_ws_ec, i)
"""

# Capture features and labels for all filepaths
print("\nFilepath", fp_ws_ec, "\nLabel", this_label, "\nPercentile Kept", this_perc_best, "\nTest Size", this_test_size, "\n")
features_ws_ec, labels_ws_ec, data_ws_ec = splitFeaturesLabels(fp_ws_ec, "asm_level", -400)

print("\nFilepath", fp_ns_ec, "\nLabel", this_label, "\nPercentile Kept", this_perc_best, "\nTest Size", this_test_size, "\n")
features_ns_ec, labels_ns_ec, data_ns_ec = splitFeaturesLabels(fp_ns_ec, "asm_level", -400)

print("\nFilepath", fp_ws_core, "\nLabel", this_label, "\nPercentile Kept", this_perc_best, "\nTest Size", this_test_size, "\n")
features_ws_core, labels_ws_core, data_ws_core = splitFeaturesLabels(fp_ws_core, "asm_level", -400)

print("\nFilepath", fp_ws_core_sc, "\nLabel", this_label, "\nPercentile Kept", this_perc_best, "\nTest Size", this_test_size, "\n")
features_ws_core_sc, labels_ws_core_sc, data_ws_core = splitFeaturesLabels(fp_ws_core_sc, "asm_level", -400)

print("\nFilepath", fp_ws_core_10, "\nLabel", this_label, "\nPercentile Kept", this_perc_best, "\nTest Size", this_test_size, "\n")
features_ws_core_10, labels_ws_core_10, data_ws_core_10 = splitFeaturesLabels(fp_ws_core_10, "asm_level", -400)

print("\nFilepath", fp_ws_sh, "\nLabel", this_label, "\nPercentile Kept", this_perc_best, "\nTest Size", this_test_size, "\n")
features_ws_sh, labels_ws_sh, data_ws_sc = splitFeaturesLabels(fp_ws_sh, "asm_level", -400)

print("\nFilepath", fp_ws_sc_sh, "\nLabel", this_label, "\nPercentile Kept", this_perc_best, "\nTest Size", this_test_size, "\n")
features_ws_sc_sh, labels_ws_sc_sh, data_ws_sc_sh = splitFeaturesLabels(fp_ws_sc_sh, "asm_level", -400)

print("\nFilepath", fp_ws_total, "\nLabel", this_label, "\nPercentile Kept", this_perc_best, "\nTest Size", this_test_size, "\n")
features_ws_total, labels_ws_total, data_ws_total = splitFeaturesLabels(fp_ws_total, "asm_level", -400)
# Sample the data

# Split data into training and testing. Make sure to stratify on the label for imbalanced data, based on the above column stats.
#X_train_ws_ec, X_test_ws_ec, y_train_ws_ec, y_test_ws_ec = train_test_split(features_ws_ec, labels_ws_ec[this_label], test_size=this_test_size, random_state=this_rand_state, stratify=labels_ws_ec[this_label])
#X_train_ns_ec, X_test_ns_ec, y_train_ns_ec, y_test_ns_ec = train_test_split(features_ns_ec, labels_ns_ec[this_label], test_size=this_test_size, random_state=this_rand_state, stratify=labels_ns_ec[this_label])
#X_train_ws_core, X_test_ws_core, y_train_ws_core, y_test_ws_core = train_test_split(features_ws_core, labels_ws_core[this_label], test_size=this_test_size, random_state=this_rand_state, stratify=labels_ws_core[this_label])
#X_train_ws_core_10, X_test_ws_core_10, y_train_ws_core_10, y_test_ws_core_10 = train_test_split(features_ws_core_10, labels_ws_core_10[this_label], test_size=this_test_size, random_state=this_rand_state, stratify=labels_ws_core_10[this_label])
#X_train_ws_sc_sh, X_test_ws_sc_sh, y_train_ws_sc_sh, y_test_ws_sc_sh = train_test_split(features_ws_sc_sh, labels_ws_sc_sh[this_label], test_size=this_test_size, random_state=this_rand_state, stratify=labels_ws_sc_sh[this_label])




# Denote an untrained, parameterized RandomForestClassifiers for the data sets. A new instance is required, as variables are not copies but pointers.
#base_rfc = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=4)
"""
AUROC, CMs, and classification scores                       Demonstrate comparison metrics of each model
chi2/importance                                             Contrasts the most dependent features in the data against those most important for each model
??rfc_ws_exclude_cloud vs rfc_muts_snps                       Demonstrate that a model built on MutS SNPs is not useful - Need AUROC, CMs, classification scores & chi2/importance
rfc_ws_exclude_cloud vs rfc_ns_exclude_cloud                Shows that the model works, but that paralog splits are more useful - Need AUROC, CMs, classification scores & chi2/importance
rfc_ws_exclude_cloud vs rfc_ws_core                         Shows that core genes are not super useful - Need AUROC, CMs, classification scores & chi2/importance
rfc_ws_exclude_cloud vs rfc_ws_core_10                      Shows that the model gains slight strength when increasing data, but there is risk of overfitting and sparsity - Need AUROC, CMs, classification scores & chi2/importance
rfc_ws_exclude_cloud vs rfc_ws_soft_core_shell              Shows that the model can still function strongly, even when completely missing the core genome - Need AUROC, CMs, classification scores & chi2/importance
rfc_ws_soft_core_shell vs rfc_ws_soft_core_shell_top_50     Shows that the model can still function strongly, even when reducing to 50 features - Need AUROC, CMs, classification scores & chi2/importance
"""
# Instantiate all RFCs, 'class_weight="balanced"' is useful for the imbalanced data. Most likely best if all identical.
rfc_muts_snps       = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
rfc_ws_ec           = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
rfc_ns_ec           = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
rfc_ws_core         = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
rfc_ws_core_sc      = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
rfc_ws_core_10      = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
rfc_ws_sh           = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
rfc_ws_sc_sh        = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
rfc_ws_sc_sh_top_50 = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")


#scoring = [ 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc' ]    # Use macro here as it is only binary. 'roc_auc' default is macro
scoring = [ 'roc_auc_ovo_weighted', 'roc_auc' ]
# Train all RFCs
skft = StratifiedKFold(n_splits=these_n_splits, random_state=this_rand_state, shuffle=True)
rfc_temp   = RandomForestClassifier(n_estimators=these_n_estimators, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, class_weight="balanced")
rfc_temp_warm   = RandomForestClassifier(n_estimators=these_n_estimators, oob_score=True, random_state=this_rand_state, n_jobs=these_n_jobs, warm_start=True)



this_out_name = "rfc_temp_skft_ws_sh_ne_" + str(these_n_estimators)
full_X = features_ws_sh.copy()
print(full_X.shape)
full_y = labels_ws_sh.copy()#[this_label]
print(full_y.shape)

print("\n\nGeneral stats for 'MDR_bin'\n")
showColStats(full_y, "MDR_bin")
print("\n\nGeneral stats for 'amiblafol_bin'\n")
showColStats(full_y, "amiblafol_bin")
print("\n\nGeneral stats for 'resistance_classes'\n")
showColStats(full_y, "resistance_classes")
print("\n\nGeneral stats for 'MDR_classes_drop_bla'\n")
showColStats(full_y, "MDR_classes_drop_bla")

# Filter to classes prevalent above a given threshold value for a given column (this_label)
dict_class_label_prevalence =  full_y[this_label].value_counts().to_dict()  # Count prevalence of each class
dict_class_label_keep_drop = { key:(True if value >=keep_threshold else False) for key, value in dict_class_label_prevalence.items() } # Comprehension to filter for 'keep_threshold'
full_y['ML_multiclass_keep'] = [ dict_class_label_keep_drop[cell] for cell in full_y[this_label].to_numpy().tolist() ]  # Add columnn to dataframe
# Index mask the dataset
#idx = full_y.index[ full_y['ML_multiclass_keep'] ]
#this_X = full_X.loc[idx,:]
this_X = full_X[ full_y['ML_multiclass_keep']].copy()
print(this_X.shape)
#this_y = full_y.loc[idx,:]
this_y = full_y[ full_y['ML_multiclass_keep'] ].copy()
print(this_y.shape)

print("\n\nGeneral stats for 'MDR_bin'\n")
showColStats(this_y, "MDR_bin")
print("\n\nGeneral stats for 'amiblafol_bin'\n")
showColStats(this_y, "amiblafol_bin")
print("\n\nGeneral stats for 'resistance_classes'\n")
showColStats(this_y, "resistance_classes")
print("\n\nGeneral stats for 'MDR_classes_drop_bla'\n")
showColStats(this_y, "MDR_classes_drop_bla")

print(type_of_target(this_y[this_label]))

#print(this_X.head())
#print(this_y[this_label].head())
# For binary or multiclass
le = preprocessing.LabelEncoder()
trans_y = list(le.fit_transform(this_y[this_label]))
this_label_trans = this_label + "_trans"
this_y[this_label_trans] = trans_y
print(trans_y == this_y[this_label].to_numpy().tolist())



X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(this_X, this_y[this_label_trans], test_size=this_test_size, random_state=this_rand_state, stratify=trans_y)
"""
# Abusive training here. NOTE deprecated as "multilabel" is not supported and would lack metrics to check
X_train_temp = this_X
X_test_temp = this_X
y_train_temp = this_y
y_test_temp = this_y
this_multi_out = MultiOutputClassifier(rfc_temp).fit(X_train_temp, y_train_temp).predict(X_test_temp)
print(this_multi_out)
"""
"""
# For multilabel
sss = StratifiedShuffleSplit(n_splits=1, test_size=this_test_size, random_state=this_rand_state)

X_train_temp = pd.DataFrame()
X_test_temp = pd.DataFrame()
y_train_temp = pd.DataFrame()
y_test_temp = pd.DataFrame()
for train_index, test_index in sss.split(this_X, this_y):
    X_train_temp, X_test_temp = this_X.iloc[train_index], this_X.iloc[test_index]
    y_train_temp, y_test_temp = this_y.iloc[train_index], this_y.iloc[test_index]

#X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(this_X, this_y, test_size=this_test_size, random_state=this_rand_state, stratify=this_y)
"""
rfc_temp_score = cross_validate( rfc_temp, X_train_temp, y_train_temp, cv=skft, scoring=scoring, n_jobs=these_n_jobs, return_estimator=True )
trained_clf = statsCrossValidate( rfc_temp_score,    "WS SHELL StratifiedKFold Shuffle TRUE", 'test_roc_auc_ovo_weighted'    )
print(rfc_temp_score)

out_test_roc = fp_to_graphic_dir + "/auroc_" + this_out_name + ".png"
out_test_cm = fp_to_graphic_dir + "/cm_" + this_out_name + ".png"
this_bar_out_name = fp_to_graphic_dir + "/bar_" + this_out_name + "_mean_importance_chi2" + ".png" 
#df_imp_feat, trained_clf = visualizeCrossValidatedClassifier( rfc_temp, rfc_temp_warm, skft, X_train_temp, y_train_temp, "CVskft graphic test AUROC ws sh", out_test_roc, "CVskft graphic test CM ws sh", out_test_cm )
df_imp_feat = visualizeCrossValidatedClassifier2( rfc_temp_score, skft, X_train_temp, y_train_temp, "CVskft graphic test AUROC ws sh", out_test_roc, "CVskft graphic test CM ws sh", out_test_cm )



#print(df_imp_feat)
df_corr = calculateCorrelationDataFrame( X_train_temp, y_train_temp, chi2, "chi2", 100 )
df_metrics = df_imp_feat.join(df_corr, how="outer")
df_metrics.sort_values(by=["mean_feature_importance_impurity", "chi2_score"] , ascending=False, inplace=True)
#print(df_metrics)
df_graphics = df_metrics.copy()
df_out_name_ftr = fp_to_graphic_dir + "/df_" + this_out_name + "_fold_importance_chi2" + ".ftr"
df_metrics.reset_index(inplace=True)
df_metrics.to_feather( df_out_name_ftr )
df_out_name_tsv = fp_to_graphic_dir + "/df_" + this_out_name + "_fold_importance_chi2" + ".tsv"
df_metrics.to_csv( df_out_name_tsv, sep="\t" )
#df_graphics = df_metrics.copy()
#choose_y_axis = "impurity_importance"
#df_graphics.sort_values(by=["mean_feature_importance_impurity", "chi2_score"] , ascending=False, inplace=True) # Note, if chi2 vals missing, then the feature from impurity importance was not recognized in the percentile threshold for the SelectPercentile calculation
df_view = df_graphics.head(num_chosen_genes)
### NOTE Insert muts variant names here at first position
df_view = pd.concat( [ df_graphics[df_graphics.index.str.startswith('group_24522')], df_view ] )
df_view = pd.concat( [ df_graphics[df_graphics.index.str.startswith('mutS')], df_view ] )
print(df_view)

imp_feat_trained_clf = list(zip(X_train_temp.columns, trained_clf.feature_importances_))
imp_feat_trained_clf.sort(reverse=True, key=lambda x: x[1])
print(imp_feat_trained_clf[:50])

y_pred_temp = trained_clf.predict(X_test_temp)
cm_disp_predictions_test = ConfusionMatrixDisplay.from_predictions(
    y_test_temp,
    y_pred_temp,
    display_labels=trained_clf.classes_,
    normalize='true'    # Normalize over rows for % of true values
)
this_ax = plt.gca()
#temp_title = 'CM for Warm RFC Classifier ' + str(len(trained_clf.estimators_))
this_ax.set_title('CM for Median RFC Classifier ' + str(len(trained_clf.estimators_)) ) 
#cm_disp_predictions.plot()
plt.show()

roc_plot_estimator = RocCurveDisplay.from_estimator(
    trained_clf,
    X_test_temp,
    y_test_temp,
    name="From Estimator on Blind Test",
    #ax=ax
)
#roc_plot_estimator.plot()
ax = plt.gca() # Get current axes
roc_plot_estimator3 = RocCurveDisplay.from_estimator(
    trained_clf,
    X_train_temp,
    y_train_temp,
    name="From Estimator on Full Training",
    ax=ax
)
#roc_plot_estimator.plot()
#ax = plt.gca() # Get current axes
ax.set_title('AUC for Median RFC Classifier ' + str(len(trained_clf.estimators_)) )
roc_plot_predictions_test = RocCurveDisplay.from_predictions(
    y_test_temp,
    y_pred_temp,
    name="From Predictions of Blind Test",
    ax=ax
)
plt.show()

# Permutation importance
"""
## With permutation importance, less biased for HIGH CARDINALITY DATA, BUT COMPUTATIONALLY EXPENSIVE. Uncomment the below lines if desired
calc_perm_imp = permutation_importance( this_clf, X_test, y_test, n_repeats=10, random_state=this_rand_state, n_jobs=3)
permutation_features = list(zip(features.columns, calc_perm_imp.importances_mean))
permutation_features.sort(reverse=True, key=lambda x: x[1])
best_permutation = [ i for i in permutation_features[:len(names_best_select)] ]    # Will be the same length as best_select, which is determined by the user indicated percentile variable "this_perc_best"
dict_best_permutation = { i[0] : i[1] for i in best_permutation }
df_best_permutation = pd.DataFrame.from_dict(dict_best_permutation, orient="index", columns=['permutation_importance'])
names_best_permutation = list(dict_best_permutation.keys())
#these_interesting_dict.append(dict_best_permutation)
#nbp_same, nbp_diff, nbp_union = checkSets([names_best_permutation])    # Takes a list of lists
#print("\nFrom Feature Importance Permutation ")
#print("Similar\n", nbp_same)
#print("Different\n", nbp_diff)
#print("Total\n", nbp_union)
"""
# Gini vs log loss
"""
# Again
# Run the classifier for important features
this_clf_important = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=4)
# Could also try RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=4, criterion="entropy", class_weight="balanced")   # Log loss not in scikit-learn 1.0.2, but in 1.1
# NOTE for the above that it did not seem to do much better or much differently, mainly gini is faster than entropy
these_important = [ i[0] for i in important_features[:num_chosen_genes] ]
X_train_important = X_train[ these_important ]
X_test_important = X_test[ these_important ]
this_clf_important.fit(X_train_important, y_train)

# Check the classifier score
y_pred_important = this_clf_important.predict(X_test_important)
scores_important = cross_val_score(this_clf_important, X_test_important, y_test, cv=5)
print("Calculated metrics for Important Features using", num_chosen_genes)
print("The below should be similar for Random Forest")
print("CrossVal", scores_important.mean())
print("OOB", this_clf_important.oob_score_)
print("Classification Report for Important Features")
print(classification_report(y_test, y_pred_important))
"""
# Capturing top 50 for subset based RFC instead of full
"""
these_important = [ i[0] for i in important_features[:num_chosen_genes] ]
X_train_important = X_train_ws_sc_sh[ these_important ]
X_test_important = X_test_ws_sc_sh[ these_important ]
rfc_ws_soft_core_shell_top_50, y_pred_important = makeRFC( rfc_ws_sc_sh_top_50, "With Split Soft Core Shell Top 50 Important", X_train_important, X_test_important, y_train_ws_sc_sh, y_test_ws_sc_sh )
"""

# Calculate classifier metrics

# Compare classifier metrics

# Visuals

####################


####################
### MAKE VISUALS
####################

#plt.figure(figsize=(8,6), dpi=100)
fig1, ax = plt.subplots(figsize=(12,8), dpi=100)
#fig1(figsize=(8,6), dpi=100)
ax2 = ax.twinx()
these_pos = np.arange(len(df_view.index.to_numpy().tolist()))
width = 0.35
ax.bar(these_pos - width/2, df_view["chi2_score"], width, label="Chi-squared Score")
ax2.bar(these_pos + width/2, df_view["mean_feature_importance_impurity"], width, label="Mean Importance", color="darkorange")
this_title = "Chi-squared Score and Mean Feature Importance for Top " + str(num_chosen_genes) + " Important Genes"
ax.set_title(this_title)
ax.set_xlabel("Gene (feature)")
ax.set_xticks( these_pos )
ax.set_xticklabels( df_view.index.to_numpy().tolist(), rotation=45, ha="right" )
ax.set_ylabel("Chi-Squared Score")
fig1.legend( loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes )
ax2.set_ylabel("Mean Importance (Gini Impurity-based)")
#ax.set_xticklabels( df_view.index.to_numpy().tolist(), rotation=45, ha="right" )
#plt.xticks = ( these_pos, df_view.index.to_numpy().tolist() )
#ax.set_xticklabels( these_pos, df_view.index.to_numpy().tolist(), rotation=45, ha="right")
plt.tight_layout()
#plt.figure(figsize=(8,6), dpi=100)
#plt.xticks(these_pos, df_view.index, rotation=45, ha="right")
#this_bar_out_name = fp_to_graphic_dir + "/bar_" + "" + "_mean_importance_chi2" + ".png" 
plt.savefig( this_bar_out_name, format='png', dpi='figure', pad_inches=0.1 )
plt.show()
"""
cm_disp_estimator = ConfusionMatrixDisplay.from_estimator(
    rfc_ws_sc_sh,
    X_test_ws_sc_sh,
    y_test_ws_sc_sh,
    display_labels=rfc_ws_sc_sh.classes_,
    normalize='true'    # Normalize over rows for % of true values
)
this_ax = plt.gca()
this_ax.set_title('RFC from With Split Soft Core Shell Estimator')
#cm_disp_estimator.plot()
#plt.show()

cm_disp_predictions = ConfusionMatrixDisplay.from_predictions(
    y_test_ws_sc_sh,
    y_pred_ws_sc_sh,
    display_labels=rfc_ws_sc_sh.classes_,
    normalize='true'    # Normalize over rows for % of true values
)
this_ax = plt.gca()
this_ax.set_title('RFC from With Split Soft Core Shell Predictions')
#cm_disp_predictions.plot()
#plt.show()
"""
"""
# Overlay AUROC comparisons
out_1 = fp_to_graphic_dir + "/auroc_" + "rfc_ws_ec_vs_rfc_ns_ec" + ".png"
compareAUROC(out_1, rfc_ws_ec, "With Split Exclude Cloud", X_test_ws_ec, y_pred_ws_ec, rfc_ns_ec, "No Split Exclude Cloud", X_test_ns_ec, y_pred_ns_ec, y_test_ws_ec)
out_2 = fp_to_graphic_dir + "/auroc_" + "rfc_ws_ec_vs_rfc_ws_core" + ".png"
compareAUROC(out_2, rfc_ws_ec, "With Split Exclude Cloud", X_test_ws_ec, y_pred_ws_ec, rfc_ws_core, "With Split Core", X_test_ws_core, y_pred_ws_core, y_test_ws_ec)
out_3 = fp_to_graphic_dir + "/auroc_" + "rfc_ws_ec_vs_rfc_ws_core_10" + ".png"
compareAUROC(out_3, rfc_ws_ec, "With Split Exclude Cloud", X_test_ws_ec, y_pred_ws_ec, rfc_ws_core_10, "With Split Core 10", X_test_ws_core_10, y_pred_ws_core_10, y_test_ws_ec)
out_4 = fp_to_graphic_dir + "/auroc_" + "rfc_ws_ec_vs_rfc_ws_sc_sh" + ".png"
compareAUROC(out_4, rfc_ws_ec, "With Split Exclude Cloud", X_test_ws_ec, y_pred_ws_ec, rfc_ws_sc_sh, "With Split Soft Core Shell", X_test_ws_sc_sh, y_pred_ws_sc_sh, y_test_ws_ec)
out_5 = fp_to_graphic_dir + "/auroc_" + "rfc_ws_sc_sh_vs_rfc_ws_sc_sh_top_50" + ".png"
compareAUROC(out_5, rfc_ws_sc_sh, "With Split Soft Core Shell", X_test_ws_sc_sh, y_pred_ws_sc_sh, rfc_ws_sc_sh_top_50, "With Split Soft Core Shell Top 50 Important", X_test_important, y_pred_important, y_test_ws_sc_sh)
"""
"""
### NOTE come back to this could be useful to include? to look up the false positives and false negatives?
# Capturing false positives and false negatives
df_rfc_ws_sc_sh_top_50 = X_test_important.copy()
these_cols = df_rfc_ws_sc_sh_top_50.columns.tolist()
these_cols.append("y_test")
df_rfc_ws_sc_sh_top_50 = df_rfc_ws_sc_sh_top_50.join(y_test_ws_sc_sh)
df_rfc_ws_sc_sh_top_50.columns = these_cols
df_rfc_ws_sc_sh_top_50["y_pred"] = y_pred_important
df_fp = df_rfc_ws_sc_sh_top_50[ (df_rfc_ws_sc_sh_top_50['y_test']==0) & (df_rfc_ws_sc_sh_top_50['y_pred']==1) ]
#print(df_fp)
df_fn = df_rfc_ws_sc_sh_top_50[ (df_rfc_ws_sc_sh_top_50['y_test']==1) & (df_rfc_ws_sc_sh_top_50['y_pred']==0) ]
#print(df_fn)
"""
##########

"""
this_lb = preprocessing.LabelBinarizer()
X_train, X_test, y_train, y_test = train_test_split(features, this_lb.fit_transform(labels['MDR_classes']), test_size=0.1, random_state=1234)
my_ovr = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=this_rand_state, n_jobs=4))
my_ovr.fit(X_train, y_train)
y_pred = my_ovr.predict(X_test)
y_pred = this_lb.inverse_transform(y_pred)
#print(my_ovr.predict_proba(X_test))
scores = cross_val_score(my_ovr, X_test, y_test, cv=5)
print(scores)
print(classification_report(y_test, y_pred))
print(labels['MDR_classes'].describe())
print(labels['MDR_classes'].value_counts())
"""

