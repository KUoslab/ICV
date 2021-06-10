import joblib
from openpyxl import load_workbook
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import csv
import time

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

clf = RandomForestRegressor(random_state=40)
clf.fit(train_X_ppr, train_y_ppr.ravel())
joblib.dump(clf, "./model/m2_55_10000")

time0 = time.time()
# Test
clf = joblib.load("./model/m2_55_10000")
pred_y_ppr = clf.predict(test_X_ppr)
pred_y = min_max_scalar.inverse_transform(pred_y_ppr.reshape(-1, 1))
print(time.time() - time0)