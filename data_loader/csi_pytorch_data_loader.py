import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import h5py
import os
from sklearn.preprocessing import LabelBinarizer
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import pandas as pd
from tqdm import tqdm


def load_data(data_path, batch_size, is_train, kwargs):
    sequence_max_len = 677
    input_feature = 270

    # 载入mat数据
    mat_data = loadmat(data_path)
    # mat_data = hdf5storage.loadmat(mat_path)
    # mat_data = h5py.File(data_path)

    # 读出csi，确定样本个数
    csi_train = mat_data["csi_train"]
    sample_count = len(csi_train)  # 训练样本个数
    csi_train = csi_train.reshape(sample_count)

    # 创建训练使用的数组，将csi训练数据复制进去
    csi_train_data = np.zeros((sample_count, 1, sequence_max_len, input_feature))
    for i in range(sample_count):
        csi_train_data[i][0] = csi_train[i].reshape((csi_train[i].shape[0], csi_train[i].shape[1])).astype(np.float)

    # 创建训练使用的标签数组，将动作名称标签复制进去（人物标签暂时不复制）
    csi_train_label = []
    for i in range(len(mat_data["csi_label"])):
        csi_train_label.append(mat_data["csi_label"][i][0][0])
    # 将训练标签数组进行one-hot编码
    # lb = LabelBinarizer()
    # csi_train_label = lb.fit_transform(csi_train_label)
    csi_train_label = pd.Categorical(csi_train_label)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(csi_train_data, csi_train_label.codes,
                                                        test_size=0.2)  # split输入的数据集必须转换成numpy类型(只能处理numpy类型的数据)

    data_loader = None
    if is_train == True:
        X_train = torch.from_numpy(X_train).float()  # numpy 转成 torch 类型
        y_train = torch.from_numpy(y_train).long()
        torch_dataset = TensorDataset(X_train, y_train)  # 训练的数据集
        data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, **kwargs,drop_last=True)
    else:
        if is_train == False:
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).long()
            torch_dataset = TensorDataset(X_test, y_test)  # 训练的数据集
            data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, **kwargs,drop_last=True)
    return data_loader


class ConvNet(nn.Module):
    def __init__(self, batch_size, sequence_max_len, input_feature):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.conv4 = nn.Conv2d(30, 40, kernel_size=5)
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(19760, 320)
        self.fc2 = nn.Linear(320, 6)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), 2))
        x = x.view(32, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    total_loss_train = 0
    criterion = nn.CrossEntropyLoss().to(device)
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.train()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss_train += loss.data
        preds = output.data.max(dim=1, keepdim=True)[1]
        correct += preds.eq(target.data.view_as(preds)).to(device).sum()
        res_i = 'Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}'.format(
            epoch, args['epochs'], batch_idx + 1, len(train_loader), loss.data
        )
    total_loss_train /= len(train_loader)
    acc = correct * 100. / len(train_loader.dataset)
    res_e = 'Epoch: [{}/{}], training loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        epoch, args['epochs'], total_loss_train, correct, len(train_loader.dataset), acc
    )
    tqdm.write(res_e)

    return model


def test(args, model, device, test_loader):
    total_loss_test = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            model.eval()
            ypred = model(data)
            loss = criterion(ypred, target)
            pred = ypred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).to(device).sum()
            total_loss_test += loss.data

        accuracy = correct * 100. / len(test_loader.dataset)
        res = 'Test: total loss: {:.6f}, correct: [{}/{}], testing accuracy: {:.4f}%'.format(
            total_loss_test, correct, len(test_loader.dataset), accuracy
        )
    tqdm.write(res)


if __name__ == '__main__':
    args = {
        'data_path': "G:/无源感知研究/数据采集/2019_07_18/实验室(3t3r)(resample)(归一化).mat",
        'kwargs': {'num_workers': 2, 'pin_memory': True},
        'batch_size': 32,
        'epochs': 100,
        'lr': 1e-3,
        'momentum': .9,
        'log_interval': 10,
        'l2_decay': 0,
        'lambda': 10,
        'sequence_max_len': 677,
        'input_feature': 270,
        'n_class': 6,
    }

    torch.backends.cudnn.benchmark = True
    train_loader = load_data(args['data_path'], args['batch_size'], is_train=True, kwargs=args['kwargs'])
    test_loader = load_data(args['data_path'], args['batch_size'], is_train=False, kwargs=args['kwargs'])

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device('cuda' if use_cuda else 'cpu')

    kwagrs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    model = ConvNet(args['batch_size'], args['sequence_max_len'], args['input_feature'])
    model.to(device)
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(
        model.parameters(),
        lr=args['lr'],
        momentum=args['momentum'],
        weight_decay=args['l2_decay']
    )

    # for epoch in range(epochs):
    for epoch in tqdm(range(1, args['epochs'] + 1)):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
