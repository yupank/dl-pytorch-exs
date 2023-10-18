import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
import torchvision.transforms as tr
import torchvision.transforms.functional as fc
from torchvision.io import read_image

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from time import time


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# cifar_train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform= tr.ToTensor())
# cifar_test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform= tr.ToTensor())
# cifar_train_loader = torch.utils.data.DataLoader(dataset= cifar_train_set, batch_size=batch_size, shuffle=True)
# cifar_test_loader = torch.utils.data.DataLoader(dataset= cifar_test_set, batch_size = 10000, shuffle= True)

# data = iter(cifar_test_loader)
# test_images, test_labels = next(data)
# print(test_images.shape)
# print(test_labels.shape)

# data = iter(cifar_train_loader)
# test_images, test_labels = next(data)
# print(test_images.shape)
# print(test_labels.shape)

# H, W = 32,32
# img = torch.randint(0,256, size=(1, H, W), dtype=torch.float32)
# img2 = torch.randint(0,256, size=(1, H, W), dtype=torch.float32)
# img1=torch.cat((img,img2),0)
# print(img1.shape)
# transforms = tr.Compose([tr.RandomResizedCrop(size=(224,224), antialias=True),
#                          tr.RandomHorizontalFlip(),
#                          tr.Normalize(mean=[0.455], std=[0.227])
#                          ])
# img = transforms(img)
# img2 = transforms(img2)
# img1=torch.cat((img,img2),0)
# print(img1.shape)


# test data just transfromed into the same size 
test_transform = tr.Compose([
    tr.Resize(size=(96,96), antialias=True),
    tr.Normalize(mean=[0.299], std=[0.137])
    ])
# transforms for data augmentation
augm_transform = (
    tr.Compose([
    tr.Resize(size=(99, 99), antialias=True),
    tr.RandomVerticalFlip(p=0.8),
    tr.RandomCrop(size=(96,96)),
    tr.Normalize(mean=[0.299], std=[0.137])
]),
tr.Compose([
    tr.Resize(size=(99, 99), antialias=True),
    tr.RandomHorizontalFlip(p=0.8),
    tr.RandomCrop(size=(96,96)),
    tr.Normalize(mean=[0.299], std=[0.137])
]),
tr.Compose([
    tr.Resize(size=(112,112), antialias=True),
    tr.RandomRotation(degrees=(3,9),expand=True),
    tr.CenterCrop(size=(96,96)),
    tr.Normalize(mean=[0.299], std=[0.137]),
    ]),
tr.Compose([
    tr.Resize(size=(112,112), antialias=True),
    tr.RandomPerspective(distortion_scale=0.15, p=0.8),
    tr.CenterCrop(size=(96,96)),
    tr.Normalize(mean=[0.299], std=[0.137]),
    ])
    )

""" utility function for data curation,
    reads the images and labels from folders and creates test and train datasets and data loaders
    train data will be augmented because 
    Args:   img_dir - the relative path to main folder with data library
            train_bs - train batch size, for the test dataser, all data will be loaded at once
            vis_inspect - number of examples to show for visual check of data augmentation,
                each example consist of test image/label with corresponding training images
    Out: test_data_loader, train_data_loader (as Torch DataLoaders)
"""
def image_loader(img_dir ='./data/STAN_patches_lbls/', train_bs = 16, vis_inspect = 0):
    label_path = img_dir + 'labels/STAN_labels.csv'
    labels_df = pd.read_csv(label_path, delimiter=',', header=0)
    test_labels = []
    train_labels = []
    test_img_tensor = torch.empty((0,5,96,96), dtype=torch.float, device=device)
    # 3D: train_img_tensor = torch.empty((0,1,5,96,96), dtype=torch.float, device=device)
    train_img_tensor = torch.empty((0,5,96,96), dtype=torch.float, device=device)
    # iterating through the folders and reading stack of monochrome images
    for idx, row in labels_df.iterrows():
        subfolder = labels_df.UID.iloc[idx]
        y_label = float(row['MUT_STATUS'])

        # the list of tensor representing a stack of 5 images(slices)
        imgs = []
        # reading the 5-layer image into tensor
        for sl in range(0,5):
            img_path = img_dir + 'patches/' + subfolder + f'/{subfolder}__sl_{sl}.png'
            img = fc.convert_image_dtype(read_image(img_path))
            imgs.append(img)
        # each 5-layer image is transformed into  [5, 96, 96] tensor to represent a single X feature
        load_img = torch.cat(imgs,dim=0)
        test_img = test_transform(load_img).unsqueeze_(0)
        # concatenating the features into a single test data tensor
        test_img_tensor = torch.cat((test_img_tensor, test_img), dim=0)
        # corresponding labels
        test_labels.append(y_label)

        # for training data augmentation, each loaded 5-layer image is cloned and transformrf into 4 different tensors
        for aug_tr in augm_transform:
            train_img = aug_tr(load_img).unsqueeze_(0)
            # concatenating the features into a single training data tensor
            # 3D: train_img_tensor = torch.cat((train_img_tensor, torch.unsqueeze(train_img,0)), dim=0)
            train_img_tensor = torch.cat((train_img_tensor, train_img), dim=0)
            # corresponding labels
            train_labels.append(y_label)
    test_label_tensor = torch.Tensor(test_labels, device=device)
    train_label_tensor = torch.Tensor(train_labels, device=device)
    
    #visual inspection
    # if idx % 15 ==0:
    #     print(f'{subfolder}_{y_label} {load_img.shape[1]}, {torch.mean(load_img)},  {torch.std(load_img)}')
    if vis_inspect > 0:
        cols = 5
        rows = vis_inspect
        shift = 13
        fig, axs = plt.subplots(rows, cols, squeeze=False, figsize=(cols*2, rows*2) )
        for idx in range(0,rows):
            show_img = fc.to_pil_image(test_img_tensor[shift+idx,0])
            axs[idx,0].imshow(np.asarray(show_img))
            axs[idx,0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[idx,0].set_title(f'test l:{test_label_tensor[shift+idx]}')
            for col in range(1,cols):
                show_img = fc.to_pil_image(train_img_tensor[(shift+idx)*4+col-1,0])
                axs[idx,col].imshow(np.asarray(show_img))
                axs[idx,col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axs[idx,col].set_title(f'train#{col-1} l: {train_label_tensor[(shift+idx)*4+col-1]}')
        plt.show()
    # data loaders to be used in the networks
    test_dataset = TensorDataset(test_img_tensor, test_label_tensor)
    train_dataset = TensorDataset(train_img_tensor, train_label_tensor)
    train_data_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size=train_bs, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size=len(test_labels), shuffle=True)
    
    return train_data_loader, test_data_loader
    # return train_img_tensor, test_img_tensor

""" CNN models """


# utility function for accuracy evaluation
def accuracy_rate(predicted, true_y):
    pred_y = [1 if pred > 0.5 else 0 for pred in predicted]
    acc_score = accuracy_score(true_y, pred_y)
    return acc_score

# utility function for false negatives evaluation - more important for the specific task than recall
def false_neg_rate(predicted, true_y):
    pred_y = [1 if pred > 0.5 else 0 for pred in predicted]
    false_neg = [pr for pr, tr in zip(pred_y, true_y) if int(tr)==1 and pr==0 ]
    return len(false_neg)/len(pred_y)

class Cnn3Layer(nn.Module):
    def __init__(self, init_nodes=16, conv_3_scale = 4, n_channel=5, conv_kernel= 5, inp_dim=28*28, drop=0.1):
        super(Cnn3Layer, self).__init__()

        self.name = f'cnn_3L{init_nodes}'
        pad = int((conv_kernel-1)/2)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(n_channel, init_nodes, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(init_nodes, init_nodes*2, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes*2),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(init_nodes*2, init_nodes*conv_3_scale, kernel_size= conv_kernel, stride=1, padding=pad),
            nn.BatchNorm2d(init_nodes*conv_3_scale),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop)

        )
        out_dim = int(inp_dim/64)
        self.lin_1 = nn.Linear(init_nodes*conv_3_scale*out_dim,1)
        self.final = nn.Sigmoid()

    def forward(self,x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = out.view(out.size(0), -1)
        out = self.lin_1(out)
        out = self.final(out)
        return out.squeeze(1)

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
        self.lin_1 = nn.Linear(init_nodes*out_dim,1)
        self.final = nn.Sigmoid()

    def forward(self,x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.lin_1(out)
        out = self.final(out)
        return out.squeeze(1)

""" A 'wrapper' function which performs model training and evaluation and reports performance metrics
    Args:   NN model, number of epochs to train, learning rate, criterion,
            train_loader - multiple batches of training data, 
            test_loader - single batch of data
    Returns: final accuracy and total execution time
"""
def cnn_image_classifier(model, train_loader, test_loader, model_tag = '',
                     epochs=10, learn_rate= 0.01, criterion = nn.BCELoss()):
    test_data = iter(test_loader)
    test_images, test_labels = next(test_data)
    # optimizer  = torch.optim.Adam(model.parameters(), lr=learn_rate)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learn_rate)
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
            ave_accuracy += accuracy_rate(tr_outputs, labels)
            loss = criterion(tr_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss +=  loss.item()
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
    false_negs = false_neg_rate(test_output, test_labels)

    return final_accur, false_negs, exec_time

in_dim = 96*96
model = Cnn3Layer(init_nodes=32, n_channel=5, conv_kernel=5, inp_dim=in_dim, drop = 0.2).to(device)

# model = Cnn4LayerSiam(init_nodes=32, n_channel=5,conv_kernel=3, inp_dim=in_dim, drop = 0.15).to(device)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f'total {params} parameters')

stan_train_loader, stan_test_loader = image_loader(train_bs=16)

# data = iter(stan_test_loader)
# test_images, test_labels = next(data)
# print(test_images.shape)
# print(test_labels.shape)

# data = iter(stan_train_loader)
# test_images, test_labels = next(data)
# print(test_images.shape)
# print(test_labels.shape)

fin_acc, false_neg, execution_tm = cnn_image_classifier(model, stan_train_loader, stan_test_loader, 
                                          model_tag='5b2', learn_rate=0.002, epochs=40)

print(f'final test accuracy: {fin_acc} false negatives: {false_neg}')
print(f'execution time: {execution_tm}')



"""
img_dir ='./data/STAN_patches_lbls/'
label_path = img_dir + 'labels/STAN_labels.csv'
labels_df = pd.read_csv(label_path, delimiter=',', header=0)

cols = 5
rows = 4
fig, axs = plt.subplots(rows, cols, squeeze=False, figsize=(cols*2, rows*2) )
for idx in range(0,4):
    subfolder = labels_df.UID.iloc[idx+1]
    imgs = []
    for sl in range(0,5):
        img_path = img_dir + 'patches/' + subfolder + f'/{subfolder}__sl_{sl}.png'
        img = fc.convert_image_dtype(read_image(img_path))
        imgs.append(img)
    # creating the tensor representing a stack of 5 images(slices)
    load_img = torch.cat(imgs,dim=0)
    print(f'{load_img.shape[1]}, {torch.mean(load_img)},  {torch.std(load_img)}')
    train_imgs = [aug_tr(load_img) for aug_tr in augm_transform]
    test_img = test_transform(load_img)
    show_img = fc.to_pil_image(test_img[0])
    axs[idx,0].imshow(np.asarray(show_img))
    axs[idx,0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axs[idx,0].set_title(f'test')
    for col in range(1,cols):
        show_img = fc.to_pil_image(train_imgs[col-1][0])
        axs[idx,col].imshow(np.asarray(show_img))
        axs[idx,col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[idx,col].set_title(f'train_{col-1}')  
    # for sl in range(0,2):
    #     show_img = fc.to_pil_image(test_img[sl])
    #     axs[idx,sl*2].imshow(np.asarray(show_img))
    #     show_img = fc.to_pil_image(train_img[sl])
    #     axs[idx,sl*2+1].imshow(np.asarray(show_img))
    #     axs[idx,sl*2].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #     axs[idx,sl*2+1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
plt.show()
"""
