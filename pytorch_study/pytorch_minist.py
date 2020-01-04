from torchvision.datasets import mnist
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

train_set = mnist.MNIST('./data', train=True)
test_set = mnist.MNIST('./data', train=False)

def data_tf(x):
    x = np.array(x,dtype='float32')/255
    x = x.reshape((-1,))  # 维度转化
    x = torch.from_numpy(x)
    return x

from torch.utils.data import DataLoader

train_data = DataLoader(train_set,batch_size=64,shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)
a,a_label = next(iter(train_data))

net = nn.Sequential(
    nn.Linear(28*28,300),
    nn.ReLU(),
    nn.Linear(300,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.data[0]
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data[0]
        acc = num_correct / im.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval()  # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.data[0]
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data[0]
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data),
                  eval_loss / len(test_data), eval_acc / len(test_data)))