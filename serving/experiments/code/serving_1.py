import os
import sys
import joblib
import numpy as np
import pandas as pd
import json
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

model_name = "random_forest"
var_packet_size = 128
var_bandwidth_tx = 408.74
var_pps_tx = 36403

model = joblib.load('./model/' + model_name)
dataset = pd.read_csv('./data/cpu_quota.csv', names=['thread_quota', 'packet_size','bandwidth_tx', 'pps_tx', 'cpu_usage'])
y = np.array(dataset['thread_quota'])
X = np.array(dataset.drop('thread_quota', axis=1))
train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.20, random_state=40)

min_max_scalar = MinMaxScaler()
train_X_ppr = min_max_scalar.fit_transform(train_X)
train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1,1))

df = pd.DataFrame(pd.Series({'packet_size':var_packet_size, 'bandwidth_tx': var_bandwidth_tx}))
df.loc['pps_tx'] = var_pps_tx
df.loc['cpu_usage'] = [60.0]
arr = np.array(df)
input_data = min_max_scalar.transform(arr)

preds = model.predict(input_data)
output_data = min_max_scalar.inverse_transform(preds.reshape(-1, 1))
cpu_quota = output_data[0][0]

print(cpu_quota)
