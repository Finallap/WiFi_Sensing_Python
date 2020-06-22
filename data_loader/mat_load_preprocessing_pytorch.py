import hdf5storage
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import h5py
from scipy.io import loadmat
import pandas as pd
from sklearn.model_selection import train_test_split

# 需要先安装，命令：pip install prefetch_generator
# https://zhuanlan.zhihu.com/p/80695364
from data_loader.DataLoaderX import DataLoaderX


def load_data(data_path, config):
    # 载入mat数据
    mat_data = loadmat(data_path)
    # mat_data = hdf5storage.loadmat(data_path)
    # mat_data = h5py.File(data_path)

    # 读出csi，确定样本个数
    csi_train = mat_data["csi_train"]
    sample_count = len(csi_train)  # 训练样本个数
    csi_train = csi_train.reshape(sample_count)

    # 创建训练使用的数组，将csi训练数据复制进去
    if (config['model_type'] == 'conv'):  # conv输入格式为4维
        csi_train_data = np.zeros((sample_count, 1, config['sequence_max_len'], config['input_feature']))
        for i in range(sample_count):
            csi_train_data[i][0] = csi_train[i].reshape((csi_train[i].shape[0], csi_train[i].shape[1])).astype(
                np.float64)
    elif (config['model_type'] == 'conv1d'):  # conv1d输入格式为3维
        csi_train_data = np.zeros((sample_count, config['input_feature'], config['sequence_max_len']))
        for i in range(sample_count):
            csi_train_data[i] = csi_train[i].reshape((csi_train[i].shape[1], csi_train[i].shape[0])).astype(np.float64)
            # csi_train_data[i] = csi_train[i].astype(np.float64)
    elif (config['model_type'] == 'lstm'):  # lstm输入格式为3维
        csi_train_data = np.zeros((sample_count, config['sequence_max_len'], config['input_feature']))
        for i in range(sample_count):
            csi_train_data[i] = csi_train[i].reshape((csi_train[i].shape[0], csi_train[i].shape[1])).astype(np.float)

    # 创建训练使用的标签数组，将动作名称标签复制进去（人物标签暂时不复制）
    csi_train_label = []
    for i in range(len(mat_data["csi_label"])):
        # csi_train_label.append(mat_data["csi_label"][i][0][0])
        csi_train_label.append(mat_data["csi_label"][i][1][0][0])
    # 将训练标签数组进行one-hot编码
    # lb = LabelBinarizer()
    # csi_train_label = lb.fit_transform(csi_train_label)
    csi_train_label = pd.Categorical(csi_train_label)

    # 划分训练集和测试集
    # split输入的数据集必须转换成numpy类型(只能处理numpy类型的数据)
    X_train, X_test, y_train, y_test = train_test_split(csi_train_data, csi_train_label.codes,
                                                        test_size=0.3)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoaderX(train_dataset, batch_size=config['batch_size'], shuffle=True,
                               drop_last=True, pin_memory=config['pin_memory'],
                               num_workers=config['num_workers'])

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoaderX(test_dataset, batch_size=config['batch_size'], shuffle=False,
                              drop_last=True, pin_memory=config['pin_memory'],
                              num_workers=config['num_workers'])

    return train_loader, test_loader, csi_train_label
