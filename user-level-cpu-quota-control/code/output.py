import csv
import sys
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import pandas as pd
# from pandas import DataFrame

# extract network throughput
with open ("output_full.txt",'r') as f:
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

# output.csv : (cpu quota, throughput)
quota = sys.argv[1]
f = open('output.csv','a',newline='')
w = csv.writer(f)
w.writerow([quota,throughput])

