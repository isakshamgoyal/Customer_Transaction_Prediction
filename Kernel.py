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
import keras
from keras.models import Sequential
from keras.layers import Dense

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
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2 ,random_state=142)


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
modelRandom = RandomForestClassifier(n_estimators=100, n_jobs=4, class_weight='balanced')
modelRandom = modelRandom.fit(X_train_res, y_train_res)
print("Results For Random Forest")
scoreRandom=modelRandom.score(X_test_res, y_test_res)
print("\nScore", scoreRandom*100)

#Boruta feature elimination
boruta_selector = BorutaPy(modelRandom, n_estimators='auto', verbose=2, max_iter=40)
boruta_selector.fit(X_train_res, y_train_res)

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
modelGaussian = GaussianNB(priors=None, var_smoothing=1e-09)
modelGaussian = modelGaussian.fit(X_train, y_train)

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
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

modelCatBoost = CatBoostClassifier(loss_function="Logloss",
                           eval_metric="AUC",
                           task_type="GPU",
                           learning_rate=0.01,
                           iterations=10000,
                           random_seed=42,
                           od_type="Iter",
                           depth=10,
                           early_stopping_rounds=500
                          )

#modelCatBoost = CatBoostClassifier(iterations=3000, learning_rate=0.03, objective="Logloss", eval_metric='AUC')

n_split = 5
kf = KFold(n_splits=n_split, random_state=42, shuffle=True)

y_valid_pred = 0 * y_train
y_test_pred = 0

for idx, (train_index, valid_index) in enumerate(kf.split(train)):
    y_train_model, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    X_train_model, X_valid = X_train.iloc[train_index,:], X_train.iloc[valid_index,:]
    _train = Pool(X_train_model, label=y_train)
    _valid = Pool(X_valid, label=y_valid)
    print( "\nFold ", idx)
    fit_model = modelCatBoost.fit(_train,
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=200,
                          plot=True
                         )
    pred = fit_model.predict_proba(X_valid)[:,1]
    print( "  auc = ", roc_auc_score(y_valid, pred) )
    y_valid_pred.iloc[valid_index] = pred
    y_test_pred += fit_model.predict_proba(test)[:,1]
    
y_test_pred /= n_split


modelCatBoost = modelCatBoost.fit(X_train, y_train)
print("Results For CatBoost")
scoreCatBoost=modelCatBoost.score(X_test, y_test)
print("\nScore", scoreCatBoost*100)
y_Pred_Cat = modelCatBoost.predict(X_train_sub)


#LightGBM
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

params = {
        'num_leaves': 8,
         'min_data_in_leaf': 42,
         'objective': 'binary',
         'max_depth': 16,
         'learning_rate': 0.0123,
         'boosting': 'gbdt',
         'bagging_freq': 5,
         'bagging_fraction': 0.8,
         'feature_fraction': 0.8201,
         'bagging_seed': 11,
         'reg_alpha': 1.728910519108444,
         'reg_lambda': 4.9847051755586085,
         'random_state': 42,
         'metric': 'auc',
         'verbosity': -1,
         'subsample': 0.81,
         'min_gain_to_split': 0.01077313523861969,
         'min_child_weight': 19.428902804238373,
         'num_threads': 4
    }

param = {
        'num_leaves': 10,
        'max_bin': 119,
        'min_data_in_leaf': 11,
        'learning_rate': 0.02,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1.0, 
        'bagging_freq': 5, 
        'feature_fraction': 0.05,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False}

param1 = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.0083,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': 1}


n_fold = 9
folds= StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=2319)

oof=np.zeros(len(X_train))
y_Pred = np.zeros(len(X_train_sub))

#folds= StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

#for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train, y_train)):
#    print('Fold', fold_n)
#    
#    X_training, X_validation = X_train.iloc[train_index], X_train.iloc[valid_index]
#    y_training, y_validation = y_train.iloc[train_index], y_train.iloc[valid_index]
#    
#    training_data = lgb.Dataset(X_training, label=y_training)
#    validation_data = lgb.Dataset(X_validation, label=y_validation)

#modelLgb = lgb.train(params,training_data,num_boost_round=20000,
#                    valid_sets = [training_data, validation_data],verbose_eval=300,early_stopping_rounds = 200)
#
#y_Pred += modelLgb.predict(X_train_sub, num_iteration=modelLgb.best_iteration)/5

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])
    
    clf = lgb.train(param1, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)
    oof[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)
    y_Pred += clf.predict(X_train_sub, num_iteration=clf.best_iteration) / folds.n_splits
    
print("CV score: {:<8.5f}".format(roc_auc_score(y_train, oof)))

# =============================================================================
# To Calculate Probability
# =============================================================================
probabilityLogistic = modelLogistic.predict_proba(X_test)[::,1]



# =============================================================================
# ANN
# =============================================================================
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 201, init = 'uniform', activation = 'relu', input_dim = 200))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 201, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 201, init = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 201, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 2000, nb_epoch = 50)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_train_sub)


for index,i in enumerate(y_pred):
    if i<0.5:
        y_pred[index] = 0
    else:
        y_pred[index] = 1        
        
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

y_Pred=y_pred.reshape(200000,)
# =============================================================================
# Creating Submission file
# =============================================================================
submission = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_Pred
    })
submission.to_csv('DataBase/submissionLGB_15.csv', index=False)


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


#91.325