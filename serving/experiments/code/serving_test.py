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
model = joblib.load('./model/' + model_name)
dataset = pd.read_csv('./data/training.csv', names=['thread_quota', 'packet_size','bandwidth_tx', 'pps_tx', 'cpu_usage'])
y = np.array(dataset['thread_quota'])
X = np.array(dataset.drop('thread_quota', axis=1))
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)

input_df = pd.read_csv('./data/input_serving.csv', names=['packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'], delimiter=',')
input_arr = np.array(input_df)
print(input_arr)

min_max_scalar = MinMaxScaler()
train_X_ppr = min_max_scalar.fit_transform(train_X)
input_data = min_max_scalar.transform(input_arr)
train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))

preds = model.predict(input_data)
output_data = min_max_scalar.inverse_transform(preds.reshape(-1, 1))

for i in range(len(output_data)):
    print(i+1, output_data[i][0])




# '''
# model_name = sys.argv[1]
# var_packet_size = int(sys.argv[2])
# # var_packet_size = 84
# var_bandwidth_tx = float(sys.argv[3])
# # var_pps_tx = math.ceil(var_bandwidth_tx/float(var_packet_size))
# var_pps_tx = 224306
# var_cpu_usage = 100.18
# '''

# model_name = "random_forest"
# var_packet_size = 128
# var_cpu_usage = 86.52
# var_bandwidth_tx = 571.39
# var_pps_tx = 52368

# model = joblib.load('./model/' + model_name)
# dataset = pd.read_csv('./data/training.csv', names=['thread_quota', 'packet_size','bandwidth_tx', 'pps_tx', 'cpu_usage'])
# y = np.array(dataset['thread_quota'])
# X = np.array(dataset.drop('thread_quota', axis=1))
# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)

# min_max_scalar = MinMaxScaler()
# train_X_ppr = min_max_scalar.fit_transform(train_X)
# # test_X_ppr = min_max_scalar.transform(test_X)
# train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))
# # test_y_ppr = min_max_scalar.transform(test_y.reshape(-1, 1))


# # df = pd.DataFrame({"packet_size":[var_packet_size], "bandwidth_tx":[var_bandwidth_tx], "pps_tx":[var_pps_tx], "cpu_usage":[var_cpu_usage]})
# # # df2 = pd.DataFrame(pd.Series({'packet_size':var_packet_size, 'bandwidth_tx': var_bandwidth_tx}))
# # # df2.loc['pps_tx'] = [var_pps_tx]
# # # df2.loc['cpu_usage'] = [var_cpu_usage]
# # arr = np.array(df)
# # input_data = min_max_scalar.transform(arr(1,-1))

# preds = model.predict(input_data)
# print(preds)
# output_data = min_max_scalar.inverse_transform(preds.reshape(-1, 1))
# cpu_quota = output_data[0][0]
# print(output_data)
# print(cpu_quota)