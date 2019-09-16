import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from sklearn.metrics import roc_curve,auc
import matplotlib.pylab as plt
from scipy import interp
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')

#Define function to draw ROC curve for each iteration through kfold cross validation, and mean ROC curve

# ROC FUNCTION BELOW IS BORROWS HEAVILY FROM FOLLOWING KERNEL:
# Title: ROC Curve with k-Fold CV
# Author: DATAI Group
# Date: NA
# Code version: NA
# Availability: https://www.kaggle.com/kanncaa1/roc-curve-with-k-fold-cv

def draw_ROC(classifier_type, X, y):
    kfold = KFold(n_splits = k, random_state = 42)
    kfold.get_n_splits(X)

    fig1 = plt.figure(figsize = [9, 9])
    ax1 = fig1.add_subplot(111, aspect = 'equal')

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 1
    for train_index, valid_index in kfold.split(X):
        #Split training / validation set at Kth fold
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        #Undersample training data
        rus = RandomUnderSampler(random_state = 42)
        X_res, y_res = rus.fit_resample(X_train, y_train)

        #Scale undersampled training set and validation set, using scale fit from training data
        scaler = StandardScaler().fit(X_res)
        X_train_scaled = scaler.transform(X_res)
        X_valid_scaled = scaler.transform(X_valid)
        
        #Fit classifier to scaled data and undersampled data
        clf = classifier_type.fit(X_train_scaled, y_res)
        
        #Predict using classifier and plot respective ROC curve
        prediction = clf.predict_proba(X_valid_scaled)
        fpr, tpr, t = roc_curve(y_valid, prediction[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw = 2, alpha = 0.3, label = 'ROC fold %d (AUC = %0.2f)' %(i, roc_auc))
        i= i+1
    
    #Calculate and plot mean ROC curve
    plt.plot([0,1],[0,1], linestyle = '--', lw = 2, color = 'black')
    mean_tpr = np.mean(tprs, axis = 0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color = 'blue',\
             label = r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw = 2, alpha = 1)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc = "lower right")
    plt.show()