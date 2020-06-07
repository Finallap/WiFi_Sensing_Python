import numpy as np
import time
import os
import threading
from tqdm import tqdm
from tqdm import trange
from multiprocessing import Process


# class Reader(threading.Thread):
class Reader(Process):
    def __init__(self, thread_no, file_list, file_dir, training_data, start_pos, end_pos):
        super(Reader, self).__init__()
        self.thread_no = thread_no
        self.file_list = file_list
        self.start_pos = int(start_pos)
        self.end_pos = int(end_pos)
        self.file_dir = file_dir
        self.training_data = training_data

    def run(self):
        for i in tqdm(range(self.start_pos, self.end_pos + 1), desc='Thread' + str(self.thread_no), mininterval=100):
            # for i in range(self.start_pos, self.end_pos + 1):
            file_path = file_dir + '/' + file_list[i]
            csv_data = np.loadtxt(file_path, dtype=np.float16, delimiter=",")
            self.training_data.append(csv_data)


class Partition(object):
    def __init__(self, file_list, thread_num):
        self.file_list = file_list
        self.block_num = thread_num

    def part(self):
        pos_list = []
        file_num = len(self.file_list)
        block_size = file_num / self.block_num
        start_pos = 0
        for i in range(self.block_num):
            if i == self.block_num - 1:
                end_pos = file_num - 1
                pos_list.append((start_pos, end_pos))
                break
            end_pos = start_pos + block_size - 1
            if end_pos >= file_num:
                end_pos = file_num - 1
            if start_pos >= file_num:
                break
            pos_list.append((start_pos, end_pos))
            start_pos = end_pos + 1

        return pos_list


if __name__ == '__main__':
    training_data = []
    # 文件名
    file_dir = '/home/shengby/Datasets/csv_generator/h750_w270_num100000'
    file_list = os.listdir(file_dir)
    # 线程数量
    # thread_nums = [32, 24, 16, 12, 8, 4, 2, 1]
    # thread_nums = [32, 24, 2, 1]
    # thread_nums = [16, 12, 8, 4]
    thread_nums = [32]

    for thread_num in thread_nums:
        training_data = []
        # 起始时间
        start_time = time.clock()
        # 划分读取部分
        p = Partition(file_list, thread_num)
        pos = p.part()
        # # 生成线程
        # t = []
        # for i in range(thread_num):
        #     t.append(Reader(i, file_list, file_dir, training_data, *pos[i]))
        # # 开启线程
        # for i in range(thread_num):
        #     t[i].start()
        # for i in range(thread_num):
        #     t[i].join()

        # 创建进程
        processes = {}
        for i in range(thread_num):
            processes[i] = Reader(i, file_list, file_dir, training_data, *pos[i])
        # 开启线程
        for i in range(thread_num):
            processes[i].start()
        for i in range(thread_num):
            processes[i].join()

        # 结束时间
        end_time = time.clock()
        print("Thread_num: %d, cost time is: %f" % (thread_num, end_time - start_time))
    print("Sum time is: %f" % time.process_time())
