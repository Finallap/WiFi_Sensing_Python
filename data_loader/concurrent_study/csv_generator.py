# -*- coding: utf-8 -*-
import numpy as np
import time
import os

if __name__ == '__main__':
    save_dir = '/home/shengby/Datasets/csv_generator/h750_w270_num100000/'
    file_nums = 100000
    for i in range(file_nums):
        file_path = save_dir + str(i) + '.csv'
        data = np.random.rand(750, 270)
        np.savetxt(file_path, data, delimiter=',')
    print(time.process_time())
