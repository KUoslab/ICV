import csv

with open('data/input.csv','w') as f:
    makewrite = csv.writer(f)
    for i in range(1000, 100000, 200):
        makewrite.writerow([i])
