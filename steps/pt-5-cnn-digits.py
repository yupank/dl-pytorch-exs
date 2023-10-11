import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
import numpy as np
from time import time

class CNNModel(nn.Module):
    def __init__(self, conv_kernel = 5, inp_dim=28*28):
        super(CNNModel, self).__init__()
        pad = int((conv_kernel-1)/2)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        out_dim = int(inp_dim/16)
        self.lin_1 = nn.Linear(32*out_dim,10)
    def forward(self,x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = out.view(out.size(0), -1)
        return self.lin_1(out)

def accuracy_rate(predicted, target):
    pred = [np.argmax(p.detach().numpy()) for p in predicted]
    correct = [i for i,j in zip(pred, target) if i==j ]
    return len(correct)/len(pred)  


# criterion = nn.CrossEntropyLoss()

""" A 'wrapper' function which performs model training and evaluation and reports performance metrics
    Args:   NN model, number of epochs to train, learning rate, criterion,
            train_loader - multiple batches of training data, 
            test_loader - single batch of data
    Returns: final accuracy and total execution time
"""
def nn_image_classifier(model, train_loader, test_loader, model_tag = '',
                     epochs=10, learn_rate= 0.01, criterion = nn.CrossEntropyLoss()):
    test_data = iter(test_loader)
    test_images, test_labels = next(test_data)
    optimizer  = torch.optim.Adam(model.parameters(), lr=learn_rate)
    start_tm = time()

    train_loss = []
    train_accuracy = []
    validation_accuracy = []
    for ep in range(epochs):  
        running_loss = 0
        ave_accuracy = 0  #epoch accuracy, averaged over batches
        test_output = model(test_images)
        validation_accuracy.append(accuracy_rate(test_output, test_labels))
        for count, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            tr_outputs = model(images)
            loss = criterion(tr_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss +=  loss.item()
            ave_accuracy += accuracy_rate(tr_outputs, labels)
        train_loss.append(running_loss/(count+1))
        train_accuracy.append(ave_accuracy/(count+1))
        print(f'loss: {running_loss} accuracy: {validation_accuracy[-1]}')
    end_tm = time()

    # reporting results
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].plot(range(1,epochs+1),train_loss)
    axs[0].set_title('loss over training cycle')

    axs[1].plot(range(1,epochs+1), train_accuracy, label="train accuracy")
    axs[1].plot(range(1,epochs+1), validation_accuracy, '--', label="test accuracy")
    axs[1].set_title('accuracy tracking')
    axs[1].legend()
    fig.suptitle(f'training of NN model {model_tag}')
    plt.savefig(f'./reports/model_{model_tag}_ep{epochs}_lr_{int(1000*learn_rate)}.svg', format='svg')
    plt.show()

    # evaluating model accuracy
    with torch.no_grad():
        test_output = model(test_images)
    torch.save(model.state_dict(), f'./models/nn_img_class_{model_tag}.pkl')
    exec_time = end_tm - start_tm
    final_accur = accuracy_rate(test_output, test_labels)

    return final_accur, exec_time

in_dim = 28*28
batch_size = 64

dig_train_set = datasets.MNIST(root='./data', train=True, download=False, transform= transforms.ToTensor())
dig_test_set = datasets.MNIST(root='./data', train=False, download=True, transform= transforms.ToTensor())
dig_train_loader = torch.utils.data.DataLoader(dataset= dig_train_set, batch_size=batch_size, shuffle=True)
dig_test_loader = torch.utils.data.DataLoader(dataset= dig_train_set, batch_size = 10000, shuffle= True)


model = CNNModel(conv_kernel=3, inp_dim=in_dim)

fin_acc, execution_tm = nn_image_classifier(model,dig_train_loader, dig_test_loader, 
                                            model_tag='conv_2L_ker_3a', learn_rate=0.007, epochs=6)
print(f'final test accuracy: {fin_acc}')
print(f'execution time: {execution_tm}')
""" NOTES: in comparison to sequential 3 linear-layer model, convolutional network shows
    longer execution time, but converges in few epochs and give accuracy > 99.5 %;
    decreasing the convolution kernel to 3 decreases execution time without affecting the accuracy
"""   