from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
import numpy as np


def mat_load_preprocessing(mat_path, input_feature):
    # 载入mat数据
    mat_data = loadmat(mat_path)

    # 读出csi，确定样本个数
    csi_train = mat_data["csi_train"]
    sample_count = len(csi_train)  # 训练样本个数
    csi_train = csi_train.reshape(sample_count)

    # 确定各个样本的长度
    sequenceLengths = []
    for i in range(sample_count):
        sequenceLengths.append(csi_train[i].shape[1])
    sequence_max_len = max(sequenceLengths)

    # 创建训练使用的数组，将csi训练数据复制进去，长度不够的进行padding
    csi_train_data = np.zeros((sample_count, sequence_max_len, input_feature))
    for i in range(sample_count):
        temp_data = np.vstack(
            (np.transpose(csi_train[i]), np.zeros((sequence_max_len - csi_train[i].shape[1], input_feature))))
        csi_train_data[i] = temp_data

    # 创建训练使用的标签数组，将动作名称标签复制进去（人物标签暂时不复制）
    csi_train_label = []
    for i in range(len(mat_data["csi_label"])):
        csi_train_label.append(mat_data["csi_label"][i][0][0])
    # 将训练标签数组进行one-hot编码
    lb = LabelBinarizer()
    csi_train_label = lb.fit_transform(csi_train_label)

    return [csi_train_data, csi_train_label]


def generator(train_data, train_label, batch_size, input_feature):
    # 确定train_data中各个样本的长度
    sample_count = len(train_data)
    sequenceLengths = []
    for i in range(sample_count):
        sequenceLengths.append(train_data[i].shape[1])

    # 将train_data中的数据按长度进行排序
    sequenceLengthSortIndex = sorted(range(len(sequenceLengths)), key=lambda k: sequenceLengths[k])
    train_data = train_data[sequenceLengthSortIndex]

    start = 0
    stop = batch_size
    while 1:
        if (start > sample_count):
            start = 0
            stop = batch_size

        batch_index = list(range(start, stop))
        batch_train_data_temp = train_data[batch_index]
        batch_train_label = train_label[batch_index]

        sequence_max_len = train_data(stop - 1).shape[1]

        # 创建训练使用的数组，将csi训练数据复制进去，长度不够的进行padding
        batch_train_data = np.zeros((batch_size, sequence_max_len, input_feature))
        for i in range(batch_size):
            temp_data = np.vstack(
                (np.transpose(batch_train_data_temp[i]),
                 np.zeros((sequence_max_len - batch_train_data_temp[i].shape[1], input_feature))))
            batch_train_data[i] = temp_data

        yield batch_train_data, batch_train_label

        start = start + batch_size
        stop = stop + batch_size
