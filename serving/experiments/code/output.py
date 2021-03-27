import csv
import sys
import numpy as np
from sklearn.metrics import mean_squared_log_error

model_name = sys.argv[1]
pkt_size = sys.argv[2]
bandwidth_tx = sys.argv[3]
quota = sys.argv[4]

# extract network throughput
with open ("data/output_full.txt",'r') as f:
    data = list(f.readlines()[6])
    flag = 0
    num = 0
    for i in range(len(data)):
        if num == 4:
            list = data[i:-4]
            break
        elif '0' <= data[i] and data[i] <= '9'and data[i+1] == ' ':
            flag = 1
            num += 1
        elif data[i] == ' ':
            if flag == 1:
                flag = 0
throughput = ''.join(list).strip()

# rmsle
rmsle = np.sqrt(mean_squared_log_error(bandwidth_tx, throughput))

# output_$model_name.csv
# [rmsle | actual network throughput | bandwidth_tx(SLO) | pkt_size | quota]
f = open('data/output'+model_name+'csv','a')
w = csv.writer(f)
w.writerow([rmsle,throughput,bandwidth_tx,pkt_size,quota])
