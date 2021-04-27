import csv
import math

with open('../data/input.csv','w', newline="") as f:
    makewrite = csv.writer(f)
    for i in range(5, 11):
        pkt_size = int(math.pow(2,i))
        if i == 6:
            interval = (350-320)/200
            tmp = 320
            for j in range(200):
                makewrite.writerow([pkt_size, tmp+interval])
                tmp += interval
        if i == 7:
            interval = (560-430)/200
            tmp = 430
            for j in range(200):
                makewrite.writerow([pkt_size, tmp+interval])
                tmp += interval
        if i == 8:
            interval = (950-320)/200
            tmp = 320
            for j in range(200):
                makewrite.writerow([pkt_size, tmp+interval])
                tmp += interval
        if i == 9:
            interval = (1750 - 380)/200
            tmp = 380
            for j in range(200):
                makewrite.writerow([pkt_size, tmp+interval])
                tmp += interval
        if i == 10:
            interval = (2640 - 400)/200
            tmp = 400
            for j in range(200):
                makewrite.writerow([pkt_size, tmp+interval])
                tmp += interval
        