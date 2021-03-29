import csv
import sys
import numpy as np
from sklearn.metrics import mean_squared_log_error

model_name = sys.argv[1]
bandwidth_tx = sys.argv[2]
tmp = sys.argv[3]

# extract network throughput
with open("./data/output_full.txt",'r') as f:
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
# rmsle = np.sqrt(mean_squared_log_error([bandwidth_tx], [throughput]))

# f = open('./data/output_'+model_name+'.csv','a')
# w = csv.writer(f)
# w.writerow([throughput,bandwidth_tx,pkt_size,quota])

wb = load_workbook('../data/result.xlsx')
ws = wb["result"]
ws.cell(tmp+2, 1).value = throughput
ws.cell(tmp+2, 2).value = bandwidth_tx
