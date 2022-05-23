import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
# print(x0)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)


x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1)).type(torch.LongTensor)
# print(y)

x,y = Variable(x), Variable(y)

print(x)
print(y)



# plt.scatter(x[:,0],x[:,1],c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = torch.sigmoid(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out = self.predict(out)
        # out = F.softmax(out)
        return out

net = Net(2,20,2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

# plt.ion()
# plt.show()

for t in range(101):
    out = net(x)
    # print(prediction)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%50 == 0:
        # plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        # print(F.softmax(prediction))

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

        prediction = torch.max(F.softmax(out, dim=1), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200  # 预测中有多少和真实值一样
        ax.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        # plt.pause(0.1)
        plt.show()


# plt.ioff()  # 停止画图sigmoid
# plt.show()