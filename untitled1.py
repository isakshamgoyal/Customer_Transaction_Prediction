# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:01:07 2019

@author: Spikee
"""

import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn import metrics
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

train = pd.read_csv("DataBase/train.csv", header='infer')
test = pd.read_csv("DataBase/test.csv", header='infer')


# =============================================================================
# Determing the Features and Labels
# =============================================================================
independent_cols =  ['ID_code','target']

X_train = train.drop(independent_cols, axis=1)
y_train = train['target']

X_train_sub= test.drop('ID_code', axis=1).copy()

# =============================================================================
# Normalizing the Columns
# =============================================================================
#from sklearn import preprocessing
#min_scaler =preprocessing.MinMaxScaler()
#X_scaled = min_scaler.fit_transform(X_train)
#X_train=pd.DataFrame(X_scaled)
#
#X_scaled_sub = min_scaler.fit_transform(X_train_sub)
#X_train_sub=pd.DataFrame(X_scaled_sub)

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X_train)
X_train=pd.DataFrame(X_scaled)

X_scaled_sub=StandardScaler().fit_transform(X_train_sub)
X_train_sub=pd.DataFrame(X_scaled_sub)
# =============================================================================
# Feature Selection
# =============================================================================
KBestModel = SelectKBest(score_func=chi2, k=100)
KBestModel = KBestModel.fit(X_train, y_train)
X_train = KBestModel.transform(X_train)
X_train_sub = KBestModel.transform(X_train_sub)

#To check if the ranks are suitable or not
colRank = []
for index,val in enumerate(KBestModel.get_support()):
    if val:
        colRank.append(KBestModel.scores_[index])

colRank.sort()

# =============================================================================
# Splitting the data into training and testing
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1 ,random_state=120, stratify=y_train)


# =============================================================================
# Oversampling- SMOTE
# =============================================================================
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
X_test_res, y_test_res = sm.fit_sample(X_test, y_test.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

# =============================================================================
# Models for Prediction
# =============================================================================
#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
modelLogistic = LogisticRegression()
modelLogistic = modelLogistic.fit(X_train_res,y_train_res)

print("Results For Logistic Regression")
scoreLogistic=modelLogistic.score(X_test_res,y_test_res)
print("\nScore", scoreLogistic*100)

y_Pred = modelLogistic.predict(X_test)

#GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
modelGaussian = GaussianNB()
modelGaussian = modelGaussian.fit(X_train, y_train)

print("Results For GaussianNB")
scoreGaussianNB=modelGaussian.score(X_test, y_test)
print("\nScore", scoreGaussianNB*100)

y_Pred = modelGaussian.predict(X_train_sub)


#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
modelRandom = RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight='balanced')
modelRandom = modelRandom.fit(X_train_res, y_train_res)
print("Results For Random Forest")
scoreRandom=modelRandom.score(X_test_res, y_test_res)
print("\nScore", scoreRandom*100)



# =============================================================================
# Creating Submission file
# =============================================================================
submission = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_Pred
    })
submission.to_csv('DataBase/submission.csv', index=False)




# =============================================================================
# Calculating the Confusion Matrix
# =============================================================================

cnf_matrixXGb = metrics.confusion_matrix(y_test, y_Pred)
print("\n",cnf_matrixXGb)

