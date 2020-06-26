from __future__ import print_function
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
import os
from torch.utils.tensorboard import SummaryWriter
import data_loader.mat_load_preprocessing_pytorch as load_csi_data
from pytorch_model.Net import buildModel

# 这个版本的acc和loss计算可能有问题

# Training settings
from pytorch_config import CONFIG

if __name__ == '__main__':
    train_loader, test_loader, _ = load_csi_data.load_data(
        os.path.join(CONFIG['dir_path'], CONFIG['data_name']), CONFIG)
    train_loader_len = len(train_loader.dataset)
    test_loader_len = len(test_loader.dataset)

    writer = SummaryWriter(CONFIG['tensorboard_log_path'])

    model = buildModel(CONFIG)
    model = model.cuda()
    print(model)

    optimizer = optim.RMSprop(model.parameters(), lr=0.001, eps=1e-2)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(1000):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.data.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / train_loader_len, train_acc / train_loader_len))
        writer.add_scalar('training loss', train_loss / train_loader_len, epoch, time.time())
        writer.add_scalar('training acc', train_acc / train_loader_len, epoch, time.time())

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = Variable(batch_x, volatile=True).cuda(), Variable(batch_y, volatile=True).cuda()
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.data.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.data.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / test_loader_len, eval_acc / test_loader_len))
        writer.add_scalar('validation loss', eval_loss / test_loader_len, epoch, time.time())
        writer.add_scalar('validation acc', eval_acc / test_loader_len, epoch, time.time())
        model.train()
