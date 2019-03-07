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

a = train.corr()['target'].sort_values(ascending=False)
col = []
sortedColNames = a.index.tolist()
for i in range(1,14):
    col.append(sortedColNames[i])
    col.append(sortedColNames[len(sortedColNames)-i])


# =============================================================================
# Determing the Features and Labels
# =============================================================================

X_train = train[col]
y_train = train['target']

#X_test= test.drop('ID_code', axis=1).copy()


#Normalizing the Columns
from sklearn import preprocessing

min_scaler =preprocessing.MinMaxScaler()
X_scaled = min_scaler.fit_transform(X_train)
X_train=pd.DataFrame(X_scaled)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25 ,random_state=0)


# =============================================================================
# Models for Prediction
# =============================================================================

from sklearn.linear_model import LogisticRegression
#modelLogistic = LogisticRegression(C=0.9, verbose=2, random_state=10, solver='newton-cg', multi_class='multinomial')

modelLogistic = LogisticRegression()

#fitting on our model
from sklearn.tree import DecisionTreeClassifier
modelDecision = DecisionTreeClassifier(random_state = 0)


from sklearn.ensemble import RandomForestClassifier
modelRandom = RandomForestClassifier()
# =============================================================================
# Fitting it into the Model
# =============================================================================
modelLogistic = modelLogistic.fit(X_train,y_train)

modelDecision = modelDecision.fit(X_train , y_train)

modelRandom = modelRandom.fit(X_train, y_train)

# =============================================================================
# Predicting the Labels
# =============================================================================
y_Pred = modelLogistic.predict(X_test)

y_Pred = modelDecision.predict(X_test)
# =============================================================================
# To Calculate Probability
# =============================================================================
probabilityLogistic = modelLogistic.predict_proba(X_test)[::,1]


# =============================================================================
# Calculating the Confusion Matrix
# =============================================================================

#cnf_matrixLogistic = metrics.confusion_matrix(y_test, predictedLogistic)
#print("\n",cnf_matrixLogistic)


# =============================================================================
# Calculating Scores
# =============================================================================

#Results for LogisticRegression
print("Results For Logistic Regression")
scoreLogistic=modelLogistic.score(X_test, y_test)
print("\nScore", scoreLogistic*100)

print("Results For Decision Tree")
scoreDecision=modelDecision.score(X_test, y_test)
print("\nScore", scoreDecision*100)

print("Results For Random Forest")
scoreRandom=modelRandom.score(X_test, y_test)
print("\nScore", scoreRandom*100)

# =============================================================================
# Plotting ROC Curve
# =============================================================================

print("ROC Curve for Logistic Regression")
fpr, tpr, _ = metrics.roc_curve(y_test, probabilityLogistic)
plt.plot(fpr, tpr)
plt.show()



