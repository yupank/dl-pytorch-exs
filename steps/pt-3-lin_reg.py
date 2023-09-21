import torch
import torch.nn as nn
from random import random
import matplotlib.pyplot as plt

"""" just an  exersize in the trining cycle using using the simplest model   """
class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self,x):
        out = self.linear(x)
        return out
    
model = LinearModel(1,1)

learnRate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learnRate)
criterion = nn.MSELoss()
x_train = torch.tensor(range(1,32), dtype=torch.float).reshape(-1,1)
y_train = torch.tensor([3*(0.9+0.2*random())*x+5+3*random() for x in x_train]).reshape(-1,1)
# y_train = torch.tensor([3*x+5 for x in x_train]).reshape(-1,1)

epochs_max = 200
loss_track = []
inputs = x_train
labels = y_train
for epoch in range(epochs_max):
    epoch += 1
    out = model(inputs)
    optimizer.zero_grad()
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    predicted = model.forward(x_train)
    loss_track.append(loss.item())
    # if epoch % 30 == 0:
    #     print(f'epoch {epoch}, loss {loss.item()}')

# showing results
fig, axs = plt.subplots(1,2, figsize=(12,6))
axs[0].plot(range(50),loss_track[:50])
axs[0].set_title('loss over training cycle')
x_vals = x_train.detach().numpy()
axs[1].plot(x_vals, predicted.detach().numpy(), label="predicted")
axs[1].plot(x_vals, y_train.detach().numpy(), 'go', label = 'data')
axs[1].set_title('model predictions')
plt.show()
print(model.state_dict())
