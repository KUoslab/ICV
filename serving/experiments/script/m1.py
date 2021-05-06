import joblib
from openpyxl import load_workbook
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import csv

'''
model = joblib.load("./model/m1_test")
dataset = pd.read_csv('./data/m1_train_test.csv', names=['packet_size', 'bandwidth_tx', 'cpu_usage'], delimiter=',')
y = np.array(dataset['cpu_usage'])
X = np.array(dataset.drop('cpu_usage', axis=1))
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)

input_df = pd.read_csv('./data/m1_input_test.csv', names=['packet_size', 'bandwidth_tx'])
input_arr = np.array(input_df)

min_max_scalar = MinMaxScaler()
train_X_ppr = min_max_scalar.fit_transform(train_X)
input_data = min_max_scalar.transform(input_arr)
train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))

preds = model.predict(input_data)
output_data = min_max_scalar.inverse_transform(preds.reshape(-1, 1))
pred_y = min_max_scalar.inverse_transform(output_data)

# for i in range(len(output_data)):
#     print(i+1, input_arr[i], ' : ', output_data[i][0], output_data[i][1])

with open('./data/m2_input_test.csv','w', newline="") as f:
    makewrite = csv.writer(f)
    for i in range(len(output_data)):
        # 'packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'
        makewrite.writerow([input_arr[i][0], input_arr[i][1], input_arr[i][1]*100, output_data[i][0]])
''' 

'''
dataset = pd.read_csv('./data/m1_train_test.csv', names=['packet_size', 'bandwidth_tx', 'cpu_usage'])
y = np.array(dataset.drop('bandwidth_tx', axis=1))
X = np.array(dataset['cpu_usage'])
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)

min_max_scalar = MinMaxScaler()
train_X_ppr = min_max_scalar.fit_transform(train_X.reshape(-1, 1))
train_y_ppr = min_max_scalar.fit_transform(train_y)

# Cross validate
n_estimators = list(range(100, 1000, 10))
max_features = [1, 2, 3, 4, 5]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

clf = RandomForestRegressor(random_state=40)
clf.fit(train_X_ppr.reshape(-1, 1), train_y_ppr)
# clf_random_cv = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=5, scoring='neg_root_mean_squared_error', verbose=2, n_jobs=-1, random_state=40)
# clf_random_cv.fit(train_X_ppr.reshape(-1, 1), train_y_ppr)

# Save model
joblib.dump(clf, "./model/m1")

# # Test
# clf = joblib.load("./model/m1")
# pred_y_ppr = clf.predict(test_X_ppr.reshape(-1, 1))
# pred_y = min_max_scalar.inverse_transform(pred_y_ppr)

# # Evaluate
# score = np.sqrt(mean_squared_log_error(test_y, pred_y))
'''



dataset = pd.read_csv('./data/m1_train.csv', names=['packet_size', 'bandwidth_tx', 'cpu_usage'])
y = np.array(dataset['cpu_usage'])
X = np.array(dataset.drop('cpu_usage', axis=1))

# Split data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)

# Preprocess data
min_max_scalar = MinMaxScaler()
train_X_ppr = min_max_scalar.fit_transform(train_X)
test_X_ppr = min_max_scalar.transform(test_X)
train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))
test_y_ppr = min_max_scalar.transform(test_y.reshape(-1, 1))

# Cross validate
n_estimators = list(range(100, 1000, 10))
max_features = [1, 2, 3, 4, 5]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

clf = RandomForestRegressor(random_state=40)
clf.fit(train_X_ppr, train_y_ppr.ravel())

# Save model
joblib.dump(clf, "./model/m1_test")

# Test
clf = joblib.load("./model/m1_test")
pred_y_ppr = clf.predict(test_X_ppr)
pred_y = min_max_scalar.inverse_transform(pred_y_ppr.reshape(-1, 1))

# Evaluate
score = np.sqrt(mean_squared_log_error(test_y, pred_y))
print(score)

for i in range(len(pred_y)):
    print(i+1, test_X[i][0], test_X[i][1], ' : ', pred_y[i])

with open('./data/m2_input_rmsle.csv','w', newline="") as f:
    makewrite = csv.writer(f)
    for i in range(len(pred_y)):
        # 'packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'
        makewrite.writerow([test_y[i], pred_y[i][0]])
