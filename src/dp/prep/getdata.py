import csv
import os

with open('datasets.txt') as datafile:
    print("Test")
    datareader = csv.reader(datafile, delimiter=',')
    for row in datareader:
        dataset = row[0]
        os.system(f'kaggle datasets download {dataset}')
