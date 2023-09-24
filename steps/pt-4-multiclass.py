import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
import numpy as np
from time import time

""" multi-class classification of hand-written digits"""

train_set = datasets.MNIST(root='./data', train=True, download=False, transform= transforms.ToTensor())
# print(f'train:  {len(train_set)} images')
# print(f'size of image: {train_set[0][0].size()}')
test_set = datasets.MNIST(root='./data', train=False, download=True, transform= transforms.ToTensor())
# print(f'test:  {len(test_set)} images')

# shallow one-layer model
class MultiLogisticModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MultiLogisticModel, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

# helper function to evaluate the accuracy
def accuracy_rate(predicted, target):
    pred = [np.argmax(p.detach().numpy()) for p in predicted]
    correct = [i for i,j in zip(pred, target) if i==j ]
    return len(correct)/len(pred)
# essential model parameters
in_dim = 28*28
out_dim = 10
batch_size = 100
epochs = 8
learn_rate = 0.002

# instantiating model
model = MultiLogisticModel(in_dim, out_dim)
criterion = nn.CrossEntropyLoss()
# instantiating the optimizer class
optimizer = torch.optim.SGD(model.parameters(), lr = learn_rate)


train_loader = torch.utils.data.DataLoader(dataset= train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset= train_set, batch_size = 10000, shuffle= True)

test_data = iter(test_loader)
test_images, test_labels = next(test_data)

# training cycle
# tracking loss and accuracy over epochs
start_tm = time()

train_loss = []
train_accuracy = []
validation_accuracy = []
for ep in range(epochs):  
    running_loss = 0
    ave_accuracy = 0  #epoch accuracy, averaged over batches
    test_accuracy = 0
    for count, (images, labels) in enumerate(train_loader):
        tr_images = images.view(-1, in_dim)
        optimizer.zero_grad()
        tr_outputs = model(tr_images)
        loss = criterion(tr_outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss +=  loss.item()
        ave_accuracy += accuracy_rate(tr_outputs, labels)
    train_loss.append(running_loss/(count+1))
    train_accuracy.append(ave_accuracy/(count+1))
    test_output = model(test_images.view(-1, in_dim))
    validation_accuracy.append(accuracy_rate(test_output, test_labels))
    print(f'loss :{running_loss}')
end_tm = time()

# reporting results
fig, axs = plt.subplots(1,2, figsize=(12,6))
axs[0].plot(range(1,epochs+1),train_loss)
axs[0].set_title('loss over training cycle')

axs[1].plot(range(1,epochs+1), train_accuracy, label="train accuracy")
axs[1].plot(range(1,epochs+1), validation_accuracy, '--', label="test accuracy")
axs[1].set_title('accuracy tracking')
axs[1].legend()
fig.suptitle('Multi-class classification model 1 (log regression)')
plt.savefig(f'./reports/model_1_ep{epochs}_btSz_{batch_size}_lr_{int(1000*learn_rate)}.svg', format='svg')
plt.show()

# evaluating model accuracy
test_output = model(test_images.view(-1, in_dim))
print(f'final test accuracy: {accuracy_rate(test_output, test_labels)}')
print(f'execution time: {end_tm - start_tm}')

torch.save(model.state_dict(), './models/mult_class_log_reg_1.pkl')

""" NOTES : 
    1) increasing of batch size decreases the accuracy and affects convergance,
        the decrease in the model perfomance can be compensated by increasing the learning rate slightly;
        this does not strongly affect the execution time;
    2) the test accuracy of the simple module has maximum about 83-86%, tipically reached after 6 - 8 epochs,
        which is not great for the MNIST dataset

"""