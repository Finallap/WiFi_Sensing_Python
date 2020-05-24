import os
import pandas as pd

import h5py
import hdf5storage
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import TensorDataset
import time
from pytorch_config import CONFIG as config

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    # data_path = os.path.join(config['dir_path'], config['data_name'])
    # data_path = 'G:\\无源感知研究\\数据采集\\2019_07_18\\实验室(3t3r).mat'
    # data_path = 'C:\\Users\\Finallap\\Desktop\\1.mat'
    data_path = '/home/shengby/Datasets/CSI_mat/Widar3/room1_20181109_2.mat'

    start = time.time()
    # mat_data = loadmat(data_path)
    mat_data = hdf5storage.loadmat(data_path)
    # mat_data = h5py.File(data_path)
    end = time.time()
    print("Finish load .mat:{} second".format(end - start))

    start = time.time()
    # 读出csi，确定样本个数
    csi_train = mat_data["csi_train"]
    sample_count = len(csi_train)  # 训练样本个数
    csi_train = csi_train.reshape(sample_count)

    # 创建训练使用的数组，将csi训练数据复制进去
    if (config['model_type'] == 'conv'):  # conv输入格式为4维
        csi_train_data = np.zeros((sample_count, 1, config['sequence_max_len'], config['input_feature']))
        for i in range(sample_count):
            csi_train_data[i][0] = csi_train[i].reshape((csi_train[i].shape[0], csi_train[i].shape[1])).astype(
                np.float)
    elif (config['model_type'] == 'conv1d'):  # conv1d输入格式为3维
        csi_train_data = np.zeros((sample_count, config['input_feature'], config['sequence_max_len']))
        for i in range(sample_count):
            csi_train_data[i] = csi_train[i].reshape((csi_train[i].shape[1], csi_train[i].shape[0])).astype(np.float)
    elif (config['model_type'] == 'lstm'):  # lstm输入格式为3维
        csi_train_data = np.zeros((sample_count, config['sequence_max_len'], config['input_feature']))
        for i in range(sample_count):
            csi_train_data[i] = csi_train[i].reshape((csi_train[i].shape[0], csi_train[i].shape[1])).astype(np.float)

    # 创建训练使用的标签数组，将动作名称标签复制进去（人物标签暂时不复制）
    csi_train_label = []
    for i in range(len(mat_data["csi_label"])):
        csi_train_label.append(mat_data["csi_label"][i][0][0])

    # 加载服务器的Widar3数据集需要这样做，否则数据类型是numpy.ndarray，而不是str
    # for i in range(len(mat_data["csi_label"])):
    #     csi_train_label[i] = csi_train_label[i][0]

    # 将训练标签数组进行one-hot编码
    # lb = LabelBinarizer()
    # csi_train_label = lb.fit_transform(csi_train_label)
    csi_train_label = pd.Categorical(csi_train_label)

    csi_train_data = torch.from_numpy(csi_train_data).float()
    csi_train_label = torch.from_numpy(csi_train_label.codes).long()
    train_dataset = TensorDataset(csi_train_data, csi_train_label)

    end = time.time()
    print("Finish processing dataset:{} second".format(end - start))

    print("pin_memory:True")
    for num_workers in range(0, 50, 5):  # 遍历worker数
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)

        start = time.time()
        for epoch in range(1, 5):
            for batch_idx, (data, target) in enumerate(train_loader):  # 不断load
                pass
        end = time.time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

    print("\npin_memory:False")
    for num_workers in range(0, 50, 5):  # 遍历worker数
        kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)

        start = time.time()
        for epoch in range(1, 5):
            for batch_idx, (data, target) in enumerate(train_loader):  # 不断load
                pass
        end = time.time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
