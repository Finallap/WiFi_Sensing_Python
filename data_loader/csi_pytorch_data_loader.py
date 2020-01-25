import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import h5py
from sklearn.preprocessing import LabelBinarizer
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import variable
import time

log_train = open('log_train.txt', 'w')
log_test = open('log_test.txt', 'w')
RESULT_TRAIN = []
RESULT_TEST = []


def load_data(data_path, batch_size, kwargs):
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
                                                        test_size=0.3)  # split输入的数据集必须转换成numpy类型(只能处理numpy类型的数据)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    torch_dataset = TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, **kwargs,
                                               drop_last=True)

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    torch_dataset = TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, **kwargs,
                                              drop_last=True)

    return train_loader, test_loader, csi_train_label


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


class CSILSTM(nn.Module):
    def __init__(self, batch_size, sequence_max_len, input_feature, hidden_size, num_class, num_layer=2):
        super().__init__()
        self.rnn = nn.LSTM(input_feature, hidden_size, num_layer)
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, inp):
        bs = inp.size()[1]
        if bs != self.bs:
            self.bs = bs
        e_out = self.e(inp)
        h0 = c0 = variable(e_out.data.new(*(self.nl, self.bs, self.hidden_size)).zero_())
        rnn_o, _ = self.rnn(e_out, (h0, c0))
        rnn_o = rnn_o[-1]
        fc = F.dropout(self.fc2(rnn_o), p=0.8)
        return F.log_softmax(fc, dim=1)

        x, = self.rnn(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


def train(args, model, device, train_loader, optimizer, epoch, writer):
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
    accuracy = correct * 100. / len(train_loader.dataset)
    res_e = 'Epoch: [{}/{}], training loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        epoch, args['epochs'], total_loss_train, correct, len(train_loader.dataset), accuracy
    )
    # TensorBoard中进行记录
    writer.add_scalar('training loss', total_loss_train, epoch, time.time())
    writer.add_scalar('training acc', accuracy, epoch, time.time())

    tqdm.write(res_e)
    log_train.write(res_e + '\n')
    RESULT_TRAIN.append([epoch, total_loss_train, accuracy])

    return model


def test(args, model, device, test_loader, epoch, writer):
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
    # TensorBoard中进行记录
    writer.add_scalar('validation loss', loss, epoch, time.time())
    writer.add_scalar('validation acc', accuracy, epoch, time.time())

    tqdm.write(res)
    RESULT_TEST.append([epoch, total_loss_test, accuracy])
    log_test.write(res + '\n')


def create_pr_curve(model, test_loader, csi_train_label):
    # helper function
    def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
        '''
        Takes in a "class_index" from 0 to 9 and plots the corresponding
        precision-recall curve
        '''
        tensorboard_preds = test_preds == class_index
        tensorboard_probs = test_probs[:, class_index]

        writer.add_pr_curve(csi_train_label.categories[class_index],
                            tensorboard_preds,
                            tensorboard_probs,
                            global_step=global_step)
        writer.close()

    # 1. gets the probability predictions in a test_size x num_classes Tensor
    # 2. gets the preds in a test_size Tensor
    # takes ~10 seconds to run
    class_probs = []
    class_preds = []
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            model.eval()
            output = model(data)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    # plot all the pr curves
    for i in range(len(csi_train_label.categories)):
        add_pr_curve_tensorboard(i, test_probs, test_preds)


if __name__ == '__main__':
    CONFIG = {
        'data_path': "G:/无源感知研究/数据采集/2019_07_18/实验室(3t3r)(resample)(归一化).mat",
        'kwargs': {'num_workers': 2, 'pin_memory': True},
        'batch_size': 32,
        'epochs': 50,
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
    train_loader, test_loader, csi_train_label = load_data(CONFIG['data_path'], CONFIG['batch_size'],
                                                           kwargs=CONFIG['kwargs'])

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device('cuda' if use_cuda else 'cpu')

    # 使用TensorBoard进行记录
    writer = SummaryWriter('tensorboard')

    model = ConvNet(CONFIG['batch_size'], CONFIG['sequence_max_len'], CONFIG['input_feature'])
    model.to(device)
    # writer.add_graph(model)

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(
        model.parameters(),
        lr=CONFIG['lr'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['l2_decay']
    )

    for epoch in tqdm(range(1, CONFIG['epochs'] + 1)):
        train(CONFIG, model, device, train_loader, optimizer, epoch, writer)
        test(CONFIG, model, device, test_loader, epoch, writer)

    create_pr_curve(model, test_loader, csi_train_label)

    torch.save(model, 'model_dann.pkl')
    writer.close()
    log_train.close()
    log_test.close()
    res_train = np.asarray(RESULT_TRAIN)
    res_test = np.asarray(RESULT_TEST)
    np.savetxt('res_train.csv', res_train, fmt='%.6f', delimiter=',')
    np.savetxt('res_test.csv', res_test, fmt='%.6f', delimiter=',')
