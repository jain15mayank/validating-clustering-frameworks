# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:39:56 2020

@author: Mayank Jain
"""

import numpy as np
import csv
import datetime

def read_data (csv_file_path):
    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if len(row) == 0:
                line_count += 1
                continue
            if line_count==0:
                timestamp = []
                house = np.zeros((27,0), dtype="float32")
                line_count += 1
            else:
                timestamp.append(datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
                temp = []
                for i in range(1, 28):
                    if row[i] == '':
                        temp.append(np.nan)
                    else:
                        temp.append(float(row[i]))
                temp = np.reshape(np.array(temp, dtype="float32"), (27, 1))
                house = np.append(house, temp, axis=1)
                line_count += 1
    #print(f'Processed {line_count} lines.')
    return (timestamp, house)