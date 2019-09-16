import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import Lasso

import warnings
warnings.filterwarnings('ignore')



#Define function to allow classifier to properly oversample only training data in kfold cross validation 

def classifier(classifier_type, X, y):
    kfold = KFold(n_splits = k, random_state = 42)
    kfold.get_n_splits(X)

    score_list = []
    best_score = 0

    conf_matrix = np.zeros((2, 2))

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
    
        #Create classifier object
        clf = classifier_type
        clf.fit(X_train_scaled, y_res)
        score = clf.score(X_valid_scaled, y_valid)
        score_list.append(score)
        
        #Set best score
        if score > best_score:
            best_train_index = train_index
            best_valid_index = valid_index
                               
        #Create confusion matrix for fold k
        conf = confusion_matrix(y_valid, clf.predict(X_valid_scaled))
    
        #Iteratively add confusion matrix to previous
        conf_matrix[0][0] += conf[0][0]
        conf_matrix[0][1] += conf[0][1]
        conf_matrix[1][0] += conf[1][0]
        conf_matrix[1][1] += conf[1][1]


    conf_clf = conf_matrix.transpose()
    conf_clf = normalize(conf_clf, axis = 0, norm = 'l1')
    conf_clf = pd.DataFrame(conf_clf, index = ['Pred \'Current\'', 'Pred \'Fail\''],
                                columns = ['Act \'Current\'', 'Act \'Fail\''])
    avg_accuracy = sum(score_list) / len(score_list)
    avg_accuracy
    
    return conf_clf, avg_accuracy