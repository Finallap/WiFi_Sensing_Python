import time
import numpy as np
import threading
import csv


def csvReader():
    with open('data.csv', 'r') as f:
        reader = csv.reader(f)


def arrayAppend(threadId):
    for i in range(60000):
        training_data.append(np.random.rand(750, 270))
        label.add(str(threadId) + '_' + str(i))


training_data = []
label = set()

if __name__ == '__main__':
    # 线程数量
    thread_num = 2
    # 起始时间
    start_time = time.clock()
    t = []
    # 生成线程
    for i in range(thread_num):
        t.append(threading.Thread(target=arrayAppend, args=(i,)))
    # 开启线程
    for i in range(thread_num):
        t[i].start()
    for i in range(thread_num):
        t[i].join()
    # 结束时间
    end_time = time.clock()
    print("Cost time is %f" % (end_time - start_time))
    print("Data size:%f" % len(training_data))
    print("Label size:%f" % len(label))
