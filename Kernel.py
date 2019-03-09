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
from sklearn import preprocessing

train = pd.read_csv("DataBase/train.csv", header='infer')
test = pd.read_csv("DataBase/test.csv", header='infer')


# =============================================================================
# Determing the Features and Labels
# =============================================================================

#a = train.corr()['target'].sort_values(ascending=False)
#col = []
#sortedColNames = a.index.tolist()
#for i in range(1,20):
#    col.append(sortedColNames[i])
#    col.append(sortedColNames[len(sortedColNames)-i])
#
#X_train = train[col]
#y_train = train['target']

independent_cols =  ['ID_code','target']

X_train = train.drop(independent_cols, axis=1)
y_train = train['target']

X_train_sub= test.drop('ID_code', axis=1).copy()

# =============================================================================
# Normalizing the Columns
# =============================================================================
min_scaler =preprocessing.MinMaxScaler()
X_scaled = min_scaler.fit_transform(X_train)
X_train=pd.DataFrame(X_scaled)
X_scaled_sub = min_scaler.fit_transform(X_train_sub)
X_train_sub=pd.DataFrame(X_scaled_sub)

# =============================================================================
# Feature Selection
# =============================================================================
KBestModel = SelectKBest(score_func=chi2, k=125)
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
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.18 ,random_state=120)


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

#Results
print("Results For Logistic Regression")
scoreLogistic=modelLogistic.score(X_test_res,y_test_res)
print("\nScore", scoreLogistic*100)
y_Pred = modelLogistic.predict(X_test)


#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
modelDecision = DecisionTreeClassifier(random_state = 0)
modelDecision = modelDecision.fit(X_train_res, y_train_res)
print("Results For Decision Tree")
scoreDecision=modelDecision.score(X_test_res, y_test_res)
print("\nScore", scoreDecision*100)
y_Pred = modelDecision.predict(X_test)


#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
modelRandom = RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight='balanced')
modelRandom = modelRandom.fit(X_train_res, y_train_res)
print("Results For Random Forest")
scoreRandom=modelRandom.score(X_test_res, y_test_res)
print("\nScore", scoreRandom*100)

#Boruta feature elimination
#boruta_selector = BorutaPy(modelRandom, n_estimators='auto', verbose=2, max_iter=40)
#boruta_selector.fit(X_train_res, y_train_res)

#XGBoost ...51.3
import xgboost as xgb
modelXgb = xgb.XGBClassifier(booster= 'gbtree', objective="binary:logistic", random_state=200)
modelXgb = modelXgb.fit(X_train, y_train)
print("Results For XGBoost")
scoreXgb=modelXgb.score(X_test, y_test)
print("\nScore", scoreXgb*100)
y_Pred = modelXgb.predict(X_test)
y_Pred = modelXgb.predict(X_train_sub)

#GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
modelGaussian = GaussianNB()
modelGaussian = modelGaussian.fit(X_train_res, y_train_res)
print("Results For GaussianNB")
scoreGaussianNB=modelGaussian.score(X_test_res, y_test_res)
print("\nScore", scoreGaussianNB*100)
y_Pred = modelGaussian.predict(X_train_sub)


#MULTINOMIAL NAIVE BAYES
from sklearn.naive_bayes import MultinomialNB
modelMultinomialNB = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
modelMultinomialNB = modelMultinomialNB.fit(X_train_res, y_train_res)
print("Results For MultinomialNB")
scoreMultinomialNB=modelMultinomialNB.score(X_test_res, y_test_res)
print("\nScore", scoreMultinomialNB*100)
y_Pred = modelMultinomialNB.predict(X_train_sub)

#CatBoost...66.8%
from catboost import CatBoostClassifier, Pool
modelCatBoost = CatBoostClassifier(loss_function="CrossEntropy",
                               eval_metric="AUC",
                               learning_rate=0.01,
                               iterations=400,
                               random_seed=42,
                               od_type="Iter",
                               depth=8,
                               border_count=32,
                               early_stopping_rounds=700,
                               task_type = "GPU",
                               logging_level='Verbose')

modelCatBoost = modelCatBoost.fit(X_train, y_train)
print("Results For CatBoost")
scoreCatBoost=modelCatBoost.score(X_test, y_test)
print("\nScore", scoreCatBoost*100)
y_Pred = modelCatBoost.predict(X_train_sub)


# =============================================================================
# To Calculate Probability
# =============================================================================
probabilityLogistic = modelLogistic.predict_proba(X_test)[::,1]


# =============================================================================
# Creating Submission file
# =============================================================================
submission = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_Pred
    })
submission.to_csv('DataBase/submission.csv', index=False)


# =============================================================================
# Plotting ROC Curve
# =============================================================================

print("ROC Curve for Logistic Regression")
fpr, tpr, _ = metrics.roc_curve(y_test, probabilityLogistic)
plt.plot(fpr, tpr)
plt.show()

# =============================================================================
# Calculating the Confusion Matrix
# =============================================================================

cnf_matrixXGb = metrics.confusion_matrix(y_test, y_Pred)
print("\n",cnf_matrixXGb)

