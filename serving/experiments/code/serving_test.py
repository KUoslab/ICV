import os
import sys
import joblib
import numpy as np
import pandas as pd
import json
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
'''
model_name = sys.argv[1]
var_packet_size = int(sys.argv[2])
# var_packet_size = 84
var_bandwidth_tx = float(sys.argv[3])
# var_pps_tx = math.ceil(var_bandwidth_tx/float(var_packet_size))
var_pps_tx = 224306
var_cpu_usage = 100.18
'''

model_name = "random_forest"
var_packet_size = 64
var_bandwidth_tx = 311.01
var_pps_tx = 27030
var_cpu_usage = 71.88


model = joblib.load('./model/' + model_name)
dataset = pd.read_csv('./data/training.csv', names=['thread_quota', 'packet_size','bandwidth_tx', 'pps_tx', 'cpu_usage'])
y = np.array(dataset['thread_quota'])
X = np.array(dataset.drop('thread_quota', axis=1))
train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.20, random_state=40)

min_max_scalar = MinMaxScaler()
train_X_ppr = min_max_scalar.fit_transform(train_X)
train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1,1))

df = pd.DataFrame(pd.Series({'packet_size':var_packet_size, 'bandwidth_tx': var_bandwidth_tx}))
df.loc['pps_tx'] = [var_pps_tx]
df.loc['cpu_usage'] = [var_cpu_usage]
arr = np.array(df)
input_data = min_max_scalar.transform(arr)

preds = model.predict(input_data)
output_data = min_max_scalar.inverse_transform(preds.reshape(-1, 1))
cpu_quota = output_data[0][0]

print(cpu_quota)
