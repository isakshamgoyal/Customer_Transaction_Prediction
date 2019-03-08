# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:01:07 2019

@author: Spikee
"""

import pandas as pd
import numpy as np

from sklearn import metrics
import matplotlib.pyplot as plt


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
from sklearn import preprocessing

min_scaler =preprocessing.MinMaxScaler()
X_scaled = min_scaler.fit_transform(X_train)
X_train=pd.DataFrame(X_scaled)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1 ,random_state=120)


X_scaled_sub = min_scaler.fit_transform(X_train_sub)
X_train_sub=pd.DataFrame(X_scaled_sub)
# =============================================================================
# Models for Prediction
# =============================================================================

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
modelLogistic = LogisticRegression()

modelLogistic = modelLogistic.fit(X_train,y_train)

#Results
print("Results For Logistic Regression")
scoreLogistic=modelLogistic.score(X_test, y_test)
print("\nScore", scoreLogistic*100)


y_Pred = modelLogistic.predict(X_test)



#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
modelDecision = DecisionTreeClassifier(random_state = 0)

modelDecision = modelDecision.fit(X_train , y_train)

print("Results For Decision Tree")
scoreDecision=modelDecision.score(X_test, y_test)
print("\nScore", scoreDecision*100)

y_Pred = modelDecision.predict(X_test)



#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
modelRandom = RandomForestClassifier()
modelRandom = modelRandom.fit(X_train, y_train)

print("Results For Random Forest")
scoreRandom=modelRandom.score(X_test, y_test)
print("\nScore", scoreRandom*100)

y_Pred = modelRandom.predict(X_test)



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

modelGaussian = modelGaussian.fit(X_train, y_train)

print("Results For GaussianNB")
scoreGaussianNB=modelGaussian.score(X_test, y_test)
print("\nScore", scoreGaussianNB*100)

y_Pred = modelGaussian.predict(X_train_sub)


#MULTINOMIAL NAIVE BAYES
from sklearn.naive_bayes import MultinomialNB
modelMultinomialNB = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

modelMultinomialNB = modelMultinomialNB.fit(X_train, y_train)

print("Results For MultinomialNB")
scoreMultinomialNB=modelMultinomialNB.score(X_test, y_test)
print("\nScore", scoreMultinomialNB*100)


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

