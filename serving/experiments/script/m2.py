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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_log_error, r2_score
import csv

'''
model = joblib.load('./model/m2_55_10000')
dataset = pd.read_csv('./data/m2_train.csv', names=['thread_quota', 'packet_size','bandwidth_tx', 'pps_tx', 'cpu_usage'])
y = np.array(dataset['thread_quota'])
X = np.array(dataset.drop('thread_quota', axis=1))
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)

input_df = pd.read_csv('./data/m2_input.csv', names=['packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'], delimiter=',')
input_arr = np.array(input_df)

min_max_scalar = MinMaxScaler()
train_X_ppr = min_max_scalar.fit_transform(train_X)
input_data = min_max_scalar.transform(input_arr)
train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))

preds = model.predict(input_data)
output_data = min_max_scalar.inverse_transform(preds.reshape(-1, 1))

with open('./data/cgroup_input_55.csv','w', newline="") as f:
    makewrite = csv.writer(f)
    for i in range(len(output_data)):
        # 'quota', 'packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'
        makewrite.writerow([int(output_data[i][0]), int(input_arr[i][0]), input_arr[i][1], input_arr[i][3]])
'''

'''
model = joblib.load('./model/random_forest')
dataset = pd.read_csv('./data/m2_train.csv', names=['thread_quota', 'packet_size','bandwidth_tx', 'pps_tx', 'cpu_usage'])
y = np.array(dataset['thread_quota'])
X = np.array(dataset.drop('thread_quota', axis=1))
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)

input_df = pd.read_csv('./data/m2_input.csv', names=['packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'], delimiter=',')
input_arr = np.array(input_df)

min_max_scalar = MinMaxScaler()
train_X_ppr = min_max_scalar.fit_transform(train_X)
input_data = min_max_scalar.transform(input_arr)
train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))

preds = model.predict(input_data)
output_data = min_max_scalar.inverse_transform(preds.reshape(-1, 1))

with open('./data/cgroup_input_test.csv','a', newline="") as f:
    makewrite = csv.writer(f)
    for i in range(len(output_data)):
        # 'quota', 'packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'
        makewrite.writerow([int(output_data[i][0]), int(input_arr[i][0]), input_arr[i][1]])
'''


'''
input_df = pd.read_csv('./data/m2_input.csv', names=['packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'], delimiter=',')
input_arr = np.array(input_df)
with open('./data/cgroup_input_test.csv','w', newline="") as f:
    makewrite = csv.writer(f)

    # slo < 300
    flag = 0
    model = joblib.load('./model/m2_10000')
    dataset = pd.read_csv('./data/m2_train_10000.csv', names=['thread_quota', 'packet_size','bandwidth_tx', 'pps_tx', 'cpu_usage'])
    y = np.array(dataset['thread_quota'])
    X = np.array(dataset.drop('thread_quota', axis=1))
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)
    min_max_scalar = MinMaxScaler()
    train_X_ppr = min_max_scalar.fit_transform(train_X)
    input_data = min_max_scalar.transform(input_arr)
    train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))
    preds = model.predict(input_data)
    output_data = min_max_scalar.inverse_transform(preds.reshape(-1, 1))

    for i in range(len(input_arr)):
        if input_arr[i][1] > 300 and flag == 0 :
            flag = 1
            model = joblib.load('./model/random_forest')
            dataset = pd.read_csv('./data/m2_train.csv', names=['thread_quota', 'packet_size','bandwidth_tx', 'pps_tx', 'cpu_usage'])
            y = np.array(dataset['thread_quota'])
            X = np.array(dataset.drop('thread_quota', axis=1))
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)
            input_df = pd.read_csv('./data/m2_input.csv', names=['packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'], delimiter=',')
            input_arr = np.array(input_df)
            min_max_scalar = MinMaxScaler()
            train_X_ppr = min_max_scalar.fit_transform(train_X)
            input_data = min_max_scalar.transform(input_arr)
            train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))
            preds = model.predict(input_data)
            output_data = min_max_scalar.inverse_transform(preds.reshape(-1, 1))

        # if input_arr[i][1] >= 1300 :
        #     output_data[i][0] += 5000
        # elif input_arr[i][1] >= 1000 :
        #     output_data[i][0] += 10000
        # elif input_arr[i][1] >= 900 :
        #     output_data[i][0] += 15000
        # elif input_arr[i][1] >= 750 :
        #     output_data[i][0] += 9000   
        # elif input_arr[i][1] >= 450 :
        #     output_data[i][0] += 5000    
        # elif input_arr[i][1] < 350 :
        #     output_data[i][0] -= 200      

        if input_arr[i][1] >= 2050 :
            output_data[i][0] += 7000
        elif input_arr[i][1] >= 1770 :
            output_data[i][0] += 5000  
        elif input_arr[i][1] >= 1600 :
            output_data[i][0] += 10000
        elif input_arr[i][1] >= 1580 :
            output_data[i][0] += 5000
        elif input_arr[i][1] >= 1330 :
            output_data[i][0] += 17000  
        elif input_arr[i][1] >= 1300 :
            output_data[i][0] += 21000
        elif input_arr[i][1] >= 1150 :
            output_data[i][0] += 20000
        elif input_arr[i][1] >= 1060 :
            output_data[i][0] += 16000  
        elif input_arr[i][1] >= 1000 :
            output_data[i][0] += 20000
        elif input_arr[i][1] >= 900 :
            output_data[i][0] += 15000
        elif input_arr[i][1] >= 750 :
            output_data[i][0] += 13000  
        elif input_arr[i][1] >= 600 :
            output_data[i][0] += 9000
        elif input_arr[i][1] >= 450 :
            output_data[i][0] += 5000
        elif input_arr[i][1] < 250 :
            output_data[i][0] -= 220

        makewrite.writerow([int(output_data[i][0]), int(input_arr[i][0]), input_arr[i][1]])
'''



dataset = pd.read_csv('./data/m2_train.csv', names=['thread_quota', 'packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'])
y = np.array(dataset['thread_quota'])
X = np.array(dataset.drop('thread_quota', axis=1))

min_max_scalar = MinMaxScaler()
# y_ppr = min_max_scalar.fit_transform(y.reshape(-1, 1))
# X_ppr = min_max_scalar.fit_transform(X)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)

train_X_ppr = min_max_scalar.fit_transform(train_X)
test_X_ppr = min_max_scalar.transform(test_X)
train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))
test_y_ppr = min_max_scalar.transform(test_y.reshape(-1, 1))

# n_estimators = list(range(100, 1000, 100))
# max_features = [1, 2, 3, 4, 5]
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf}

# clf = RandomForestRegressor(random_state=40)
# clf_random_cv = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=5, scoring='neg_root_mean_squared_error', verbose=2, n_jobs=-1, random_state=40)
# clf_random_cv.fit(train_X_ppr, train_y_ppr.ravel())
# joblib.dump(clf_random_cv, "./model/m2_55")

clf = RandomForestRegressor(random_state=40)
clf.fit(train_X_ppr, train_y_ppr.ravel())
joblib.dump(clf, "./model/m2_55_10000")

clf = joblib.load("./model/m2_55_10000")
pred_y_ppr = clf.predict(test_X_ppr)
pred_y = min_max_scalar.inverse_transform(pred_y_ppr.reshape(-1, 1))

for i in range(len(test_y)):
    print(test_y[i], pred_y[i])

score = np.sqrt(mean_squared_log_error(test_y, pred_y))
print(score)

with open('./data/m2_rmsle.csv','w', newline="") as f:
    makewrite = csv.writer(f)
    for i in range(len(pred_y)):
        # 'packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'
        makewrite.writerow([test_y[i], pred_y[i][0]])