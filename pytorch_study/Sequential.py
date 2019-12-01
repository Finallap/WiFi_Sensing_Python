# _*_coding:utf-8_*_
import torch
from torch.autograd import Variable

# 批量输入的数据量
batch_n = 100
# 通过隐藏层后输出的特征数
hidden_layer = 100
# 输入数据的特征个数
input_data = 1000
# 最后输出的分类结果数
output_data = 10

x = Variable(torch.randn(batch_n, input_data), requires_grad=False)
y = Variable(torch.randn(batch_n, output_data), requires_grad=False)

models = torch.nn.Sequential(
    # 首先通过其完成从输入层到隐藏层的线性变换
    torch.nn.Linear(input_data, hidden_layer),
    # 经过激活函数
    torch.nn.ReLU(),
    # 最后完成从隐藏层到输出层的线性变换
    torch.nn.Linear(hidden_layer, output_data)
)
print(models)

if torch.cuda.is_available():
    models.cuda()

epoch_n = 301
learning_rate = 1e-4
loss_fn = torch.nn.MSELoss()
optimzer = torch.optim.Adam(models.parameters(),lr = learning_rate)

for epoch in range(epoch_n):
    y_pred = models(x)
    loss = loss_fn(y_pred, y)
    print("Epoch:{},Loss:{:.4f}".format(epoch, loss.data))
    optimzer.zero_grad()

    loss.backward()

    optimzer.step()