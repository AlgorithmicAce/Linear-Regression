import torch
from torch import optim, nn
import matplotlib.pyplot as plt

X = torch.arange(-3, 3, 0.1).view(-1,1)
F = -3 * X + 1
Y = F + 0.5 * torch.randn(X.shape)

class LinearRegression(nn.Module):
    def __init__(self, din, dout):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(din, dout)
    
    def forward(self,x):
        y = self.linear1(x)
        return y

model = LinearRegression(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

loss_list = []
n_epoch = 20

for epoch in range(n_epoch):
    yhat = model(X)
    loss = criterion(yhat, Y)
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

eq = model(X)
plt.plot(X, Y, 'ro', label = 'Datapoints')
plt.plot(X, eq.detach(), label = 'Model Line')
plt.show()

plt.plot(loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss against Epoch graph')
plt.show()
print("The predicted bias for the model is", model.linear1.weight.detach().item())
print("The predicted weight for the model is", model.linear1.bias.detach().item())
print(loss_list[-1])