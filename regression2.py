import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt



xx = torch.rand(100000, 1)
f = xx*2+1   # mu
g = 0.7-0.5*xx   # sigma
yy = torch.normal(f, g)



class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict1 = torch.nn.Linear(n_hidden, n_output)   # output layer
        self.soft1 = torch.nn.Softsign()
        self.hidden2 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict2 = torch.nn.Linear(n_hidden, n_output)  # output layer
        self.soft2 = torch.nn.Softsign()

        # self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # hidden layer
        # self.soft = torch.nn.Softsign()
        # self.predict = torch.nn.Linear(n_hidden, n_output, bias=True)  # output layer

    def forward(self, x):
        x1 = self.hidden1(x)
        x1 = self.soft1(x1)
        x1 = self.predict1(x1)             # for mu

        x2 = self.hidden2(x)
        x2 = self.soft2(x2)
        x2 = self.predict2(x2)             # for sigma

        return x1, x2

        # x = self.hidden(x)
        # x = self.soft(x)
        # x = self.predict(x)
        #
        # return x

net = Net(n_feature=1, n_hidden=32, n_output=1)     # define the network
print(net)

optimizer = optim.Adam(net.parameters(), lr=0.1)
# loss_func = torch.nn.MSELoss()

loader = DataLoader(dataset=(xx, yy), batch_size=128)

for t in range(250):

    for batch_idx, (x, y) in enumerate(loader):

        predMean, predVar = net.forward(x)

        # result = net.forward(x)     # input x and predict based on x
        # predMean, predVar = result.split(1, 1)   # 在维度1（后一个1）上进行间隔为1的拆分

        # print(predMean.data.numpy())
        # print(predVar.data.numpy())

        loss = torch.mean((predMean - y).pow(2) + (predVar - (predMean - y).pow(2)).pow(2))
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


xs = xx[:500]

predMean, predVar = net.forward(xs)

# result = net.forward(xs)
# predMean, predVar = result.split(1, 1)

# print(predMean.data.numpy())
# print(predVar.data.numpy())


plt.subplot(121)
plt.scatter(xs.data.numpy(), predMean.data.numpy(), label='predict mean', color='b')
plt.plot(xs.data.numpy(), f[:500].data.numpy(), label='mean', color='r')
plt.legend()


plt.subplot(122)
plt.scatter(xs.data.numpy(), np.sqrt(predVar.data.numpy()), label='predict std', color='b')
plt.plot(xs.data.numpy(), g[:500].data.numpy(), label='std', color='r')
plt.legend()


plt.show()



