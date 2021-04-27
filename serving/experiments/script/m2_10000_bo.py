import os
import sys
import joblib
import numpy as np
import pandas as pd
import json
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_log_error, r2_score, mean_squared_error
from bayes_opt import BayesianOptimization
import xgboost as xgb
import lightgbm as lgb
import csv

'''
model = joblib.load("./model/m2_10000_bo")
dataset = pd.read_csv('./data/m2_train.csv', names=['thread_quota', 'packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'])
y = np.array(dataset['thread_quota'])
X = np.array(dataset.drop('thread_quota', axis=1))
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)

input_df = pd.read_csv('./data/m2_input.csv', names=['packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'])
input_arr = np.array(input_df)

min_max_scalar = MinMaxScaler()
train_X_ppr = min_max_scalar.fit_transform(train_X)
input_data = min_max_scalar.transform(input_arr)
train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))

preds = model.predict(input_data)
output_data = min_max_scalar.inverse_transform(preds.reshape(-1, 1))

# for i in range(len(output_data)):
#     print(i+1, input_arr[i][0], input_arr[i][1], input_arr[i][2], input_arr[i][3], ' : ', output_data[i])

with open('./data/cgroup_input_test.csv','w', newline="") as f:
    makewrite = csv.writer(f)
    for i in range(len(output_data)):
        # 'IO cpu', 'packet_size', 'bandwidth_tx'
        makewrite.writerow([int(output_data[i][0]), input_arr[i][0], input_arr[i][1]])

for i in range(len(input_arr)):
    print(output_data[i])
'''


def mean_absolute_percentage_error(test_y, pred_y):
    test_y, pred_y = np.array(test_y), np.array(pred_y)
    return np.mean(np.abs((test_y - pred_y)/test_y))*100
def XGB_cv(max_depth, learning_rate, n_estimators, gamma, 
            min_child_weight, subsample, colsample_bytree, silent=True, nthread=-1):
    model = xgb.XGBRegressor(max_depth=int(max_depth),
                            learning_rate=learning_rate,
                            n_estimators=int(n_estimators),
                            gamma=gamma,
                            min_child_weight=min_child_weight,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            nthread=nthread
                            )
    model.fit(train_X_ppr, train_y_ppr.reshape(-1, 1))
    
    pred_y_ppr = model.predict(test_X_ppr)
    pred_y = min_max_scalar.inverse_transform(pred_y_ppr.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(test_y, pred_y))
    rmsle = np.sqrt(mean_squared_log_error(test_y, pred_y))
    r2 = r2_score(test_y, pred_y)
    mape = mean_absolute_percentage_error(test_y, pred_y)

    # for i in range(len(test_y)):
    #     print(test_y[i], pred_y[i])
    # print(rmsle)

    return r2
def rf_evaluate(colsample_bytree, gamma, learning_rate, max_depth, min_child_weight, n_estimators, subsample):
    model = xgb.XGBRegressor(max_depth=int(max_depth),
                            learning_rate=learning_rate,
                            n_estimators=int(n_estimators),
                            gamma=gamma,
                            min_child_weight=min_child_weight,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            )
    return model

dataset = pd.read_csv('./data/m2_10000.csv', names=['thread_quota', 'packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'])
y = np.array(dataset['thread_quota'])
X = np.array(dataset.drop('thread_quota', axis=1))

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)

min_max_scalar = MinMaxScaler()
train_X_ppr = min_max_scalar.fit_transform(train_X)
test_X_ppr = min_max_scalar.transform(test_X)
train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))
test_y_ppr = min_max_scalar.transform(test_y.reshape(-1, 1))

pbounds = {'max_depth' : (3, 7),
            'learning_rate' : (0.01, 0.2),
            'n_estimators' : (5000, 10000),
            'gamma' : (0, 100),
            'min_child_weight' : (0, 3),
            'subsample' : (0.5, 1),
            'colsample_bytree' : (0.2, 1)
            }
bo = BayesianOptimization(f=XGB_cv, pbounds=pbounds, verbose=2, random_state=1)
bo.maximize(init_points=2, n_iter=10, acq='ei', xi=0.01)
print(bo.max)

params = bo.max['params']
model = rf_evaluate(colsample_bytree = params['colsample_bytree'],
                    gamma = params['gamma'],
                    learning_rate = params['learning_rate'],
                    max_depth = params['max_depth'],
                    min_child_weight = params['min_child_weight'],
                    n_estimators = params['n_estimators'],
                    subsample = params['subsample']
                    )
model.fit(train_X_ppr, train_y_ppr.reshape(-1, 1))
joblib.dump(model, "./model/m2_10000_bo")


'''
def lgb_evaluate(numLeaves, maxDepth, scaleWeight, minChildWeight, subsample, colSam, output = 'score'):
    reg=lgb.LGBMRegressor(num_leaves=31, max_depth= 2,scale_pos_weight= scaleWeight, min_child_weight= minChildWeight, subsample= 0.4, colsample_bytree= 0.4, learning_rate=0.05,   n_estimators=20)
    scores = cross_val_score(reg, train_X, train_y, cv=5, scoring='neg_mean_squared_log_error')

    if output == 'score' :
        return np.mean(scores)
    if output == 'model' :
        return reg

def bayesOpt(train_X, train_y) : 
    lgbBO = BayesianOptimization(lgb_evaluate, {'numLeaves':(5, 90), 'maxDepth':(2, 90), 'scaleWeight':(1, 10000), 'minChildWeight':(0.01, 70), 'subsample':(0.4, 1), 'colSam':(0.4, 1)})
    lgbBO.maximize(init_points=5, n_iter=30)
    print(lgbBO.res)
    return lgbBO


dataset = pd.read_csv('./data/m2_10000.csv', names=['thread_quota', 'packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'])
y = np.array(dataset['thread_quota'])
X = np.array(dataset.drop('thread_quota', axis=1))

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)

min_max_scalar = MinMaxScaler()
train_X_ppr = min_max_scalar.fit_transform(train_X)
test_X_ppr = min_max_scalar.transform(test_X)
train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))
test_y_ppr = min_max_scalar.transform(test_y.reshape(-1, 1))

lgbBO = bayesOpt(train_X_ppr, train_y_ppr.reshape(-1, 1))
params = lgbBO.max['params']
model = lgb_evaluate(
            numLeaves=params['numLeaves'],
            maxDepth=params['maxDepth'],
            scaleWeight=params['scaleWeight'],
            minChildWeight=params['minChildWeight'],
            subsample=params['subsample'],
            colSam=params['colSam'],
            output='model'
        )
model.fit(test_X_ppr, test_y_ppr.ravel())
joblib.dump(model, "./model/m2_10000_bo")

# with open('./data/m2_input_test.csv','w', newline="") as f:
#     makewrite = csv.writer(f)
#     for i in range(len(test_y)):
#         # 'IO cpu', 'packet_size', 'bandwidth_tx'
#         makewrite.writerow([test_y[i], test_X[i][0], test_X[i][1], test_X[i][2], test_X[i][3]])
'''