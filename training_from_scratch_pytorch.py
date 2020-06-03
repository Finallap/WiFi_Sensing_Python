from __future__ import print_function
import time

import torch
import torch.optim as optim
import os

from torch.utils.tensorboard import SummaryWriter
import data_loader.mat_load_preprocessing_pytorch as load_csi_data

# Training settings
from pytorch_config import CONFIG
from pytorch_model.Net import buildModel


def train(model, device, train_loader, optimizer, epoch, writer):
    train_loss = 0.
    train_acc = 0.
    loss_func = torch.nn.CrossEntropyLoss()
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.data.item()
        # 计算分类的准确率
        _, pred = output.max(1)
        num_correct = (pred == target).sum().item()
        acc = num_correct / data.shape[0]
        train_acc += acc
    print('Epoch: {},Train Loss: {:.6f}, Acc: {:.6f}'.format(epoch, train_loss / len(train_loader),
                                                             train_acc / len(train_loader)))
    writer.add_scalar('training loss', train_loss / len(train_loader), epoch, time.time())
    writer.add_scalar('training acc', train_acc / len(train_loader), epoch, time.time())


def eval(model, device, test_loader, epoch, writer):
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    test_loss = 0.
    val_acc = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = loss_func(output, target)
            test_loss += loss.data.item()
            # 计算分类的准确率
            _, pred = output.max(1)
            num_correct = (pred == target).sum().item()
            acc = num_correct / data.shape[0]
            val_acc += acc
        print('Epoch: {}, Validation Loss: {:.6f}, Acc: {:.6f}'.format(epoch, test_loss / len(test_loader),
                                                                       val_acc / len(test_loader)))
        writer.add_scalar('validation loss', test_loss / len(test_loader), epoch, time.time())
        writer.add_scalar('validation acc', val_acc / len(test_loader), epoch, time.time())
    return val_acc / len(test_loader)


def main():
    use_cuda = torch.cuda.is_available()
    random_seed = 1
    torch.manual_seed(random_seed)
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_loader, test_loader, _ = load_csi_data.load_data(
        os.path.join(CONFIG['dir_path'], CONFIG['data_name']), CONFIG)
    train_loader_len = len(train_loader.dataset)
    test_loader_len = len(test_loader.dataset)

    writer = SummaryWriter(CONFIG['tensorboard_log_path'])

    model = buildModel(CONFIG)
    model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'])
    optimizer = optim.RMSprop(model.parameters(), lr=CONFIG['lr'], eps=1e-2)

    min_val_loss = float("inf")
    for epoch in range(CONFIG['epochs']):
        train(model, device, train_loader, optimizer, epoch, writer)
        val_loss = eval(model, device, test_loader, epoch, writer)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model, CONFIG['model_save_path'])


if __name__ == '__main__':
    main()
