import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
import numpy as np
from time import time

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

class Cnn3Layer(nn.Module):
    def __init__(self, init_nodes=16, conv_3_scale = 4, n_channel=1, conv_kernel= 5, inp_dim=28*28, drop=0.1):
        super(Cnn3Layer, self).__init__()

        self.name = f'cnn_3L{init_nodes}'
        pad = int((conv_kernel-1)/2)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(n_channel, init_nodes, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(init_nodes, init_nodes*2, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(init_nodes*2, init_nodes*conv_3_scale, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes*conv_3_scale),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)

        )
        out_dim = int(inp_dim/64)
        self.lin_1 = nn.Linear(init_nodes*conv_3_scale*out_dim,10)

    def forward(self,x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = out.view(out.size(0), -1)
        return self.lin_1(out)

class Cnn4LayerSiam(nn.Module):
    def __init__(self, init_nodes=8, n_channel=3, conv_kernel= 5, inp_dim=28*28, drop= 0.1):
        super(Cnn4LayerSiam, self).__init__()
        self.name = f'cnn_4LS{init_nodes}'
        pad = int((conv_kernel-1)/2)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(n_channel, init_nodes, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(init_nodes, init_nodes*2, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(init_nodes*2, init_nodes*2, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(init_nodes*2, init_nodes, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        out_dim = int(inp_dim/256)
        # self.lin_1 = nn.Sequential(nn.Linear(init_nodes*out_dim,10), nn.Softmax(dim=10))
        self.lin_1 = nn.Linear(init_nodes*out_dim,10)

    def forward(self,x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        # out = out.view(out.size(0), -1)
        return self.lin_1(out)

# utility function
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
        with torch.no_grad():
            test_output = model(test_images.to(device))
        validation_accuracy.append(accuracy_rate(test_output, test_labels))
        for count, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
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
    fig.suptitle(f'training of {model.name} model {model_tag}')
    plt.savefig(f'./reports/{model.name}_{model_tag}_ep{epochs}_lr_{int(1000*learn_rate)}.svg', format='svg')
    plt.show()
    torch.save(model.state_dict(), f'./models/{model.name}_{model_tag}.pkl')
    # evaluating model accuracy
    with torch.no_grad():
        test_output = model(test_images)
    
    exec_time = end_tm - start_tm
    final_accur = accuracy_rate(test_output, test_labels)

    return final_accur, exec_time



batch_size = 64
# MNIST digits classification
dig_train_set = datasets.MNIST(root='./data', train=True, download=False, transform= transforms.ToTensor())
dig_test_set = datasets.MNIST(root='./data', train=False, download=True, transform= transforms.ToTensor())
dig_train_loader = torch.utils.data.DataLoader(dataset= dig_train_set, batch_size=batch_size, shuffle=True)
dig_test_loader = torch.utils.data.DataLoader(dataset= dig_test_set, batch_size = 10000, shuffle= True)

# in_dim = 28*28
# model = CNNModel(conv_kernel=3, inp_dim=in_dim)
# fin_acc, execution_tm = nn_image_classifier(model,dig_train_loader, dig_test_loader, 
#                                             model_tag='conv_2L_ker_3a', learn_rate=0.01, epochs=8)

# CIFAR10 images classification
cifar_train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform= transforms.ToTensor())
cifar_test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform= transforms.ToTensor())
cifar_train_loader = torch.utils.data.DataLoader(dataset= cifar_train_set, batch_size=batch_size, shuffle=True)
cifar_test_loader = torch.utils.data.DataLoader(dataset= cifar_test_set, batch_size = 10000, shuffle= True)

in_dim = 32*32
# model = Cnn3Layer(init_nodes=16, n_channel=3, conv_kernel=3, inp_dim=in_dim, drop = 0.15).to(device)
model = Cnn4LayerSiam(init_nodes=16, n_channel=3,conv_kernel=5, inp_dim=in_dim, drop = 0.1).to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f'total {params} parameters')
fin_acc, execution_tm = nn_image_classifier(model,cifar_train_loader, cifar_test_loader, 
                                          model_tag='3a', learn_rate=0.005, epochs=6)

print(f'final test accuracy: {fin_acc}')
print(f'execution time: {execution_tm}')
""" NOTES: 
    1) digits classification task: 
        in comparison to sequential 3 linear-layer model, 2 convolutional-layer network shows
        longer execution time, but converges in few epochs and give accuracy about 99 %;
        decreasing the convolution kernel to 3 decreases execution time without affecting the accuracy
    2) CIFAR10 images classification task:
        2 convolutional-layer network converged quickly (within 5 epochs) but initial accuracy was about 69%, 
        doubling the nodes numbers did not increase accuracy with obvious increase of execution time,
        introduction of dropout layer decreased the gap between the train and test accuracy (i.e. reduced over-training) 
        but slowed down the training slightly; nn.Dropout2d was worse in this regard then Dropout()
    3)  introduction of 3rd conv layer improved accuracy (to ~ 75% ) and convergence,
        increasing the number of notes increased performance (slightly);
    4) 4-layer "siamese" CNN improved the accuracy further, accuracy reached > 77% after 25-30 min training (on CPU) 
"""   