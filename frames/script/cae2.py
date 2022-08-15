import copy
import torch
import time

import numpy as np
import matplotlib.pyplot as plt
import torchvision
from tqdm import trange

from util import imshow, plot_sta
from data_process import load_data, write_data

# data_dir = 'Cat_Dog_data/train'
data_dir = 'data' # load from Kaggle

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((350,350)),
                                torchvision.transforms.ToTensor()
                               ])# TODO: compose transforms here
dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform) # TODO: create the ImageFolder
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True) # TODO: use the ImageFolder dataset to create the DataLoader


inputs= iter(dataloader)

# Get a batch of training data
inputs, classes = next(iter(dataloader))
inputs = inputs[0:4]
classes = classes[0:4]
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, classes)

device = torch.device("cuda")
print(torch.cuda.is_available())

dataiter = iter(dataloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # 86
            nn.ReLU(True), #
            nn.MaxPool2d(kernel_size=3, stride=2), # 42
            nn.Conv2d(64, 192, kernel_size=5, padding=2),# 42
            nn.ReLU(True),#
            nn.MaxPool2d(kernel_size=3, stride=2), # 20
            nn.Conv2d(192, 384, kernel_size=3, padding=1),# 20
            nn.ReLU(True),#
            nn.Conv2d(384, 256, kernel_size=3, padding=1),# 20
            nn.ReLU(True),#
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# 20
            nn.ReLU(True),# 20
            nn.MaxPool2d(kernel_size=3, stride=2), #
            nn.AdaptiveAvgPool2d((2, 2)), # 2
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'), # 8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, (3,3)), # 10
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 384, (3,3), padding=1), # 10
            nn.ReLU(True),
            nn.ConvTranspose2d(384, 192, (3,3), padding=1), # 10
            nn.Upsample(scale_factor=2, mode='nearest'), # 20
            nn.ReLU(True),
            nn.ConvTranspose2d(192, 64, (5,5), stride = 2),  # 43
            nn.Upsample(scale_factor=2, mode='nearest'), # 86
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, (12,12), stride=4, padding=1),  # 350  
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

net = Autoencoder().to(device)


import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=1e-3)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

sta = {'train':{'epoch_loss':[]},'dev':{'epoch_loss':[]}}

para_dict = {}

best_loss = 50
num_epochs = 10

for epoch in trange(num_epochs):
    for phase in ['train', 'dev']:
        if phase == 'train':
            net.train()  # Set model to training mode
        else:
            net.eval()   # Set model to evaluate mode

        running_loss = 0.0

        for batch in dataloader:
            
            inputs = batch[0].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)
                loss = criterion(outputs, inputs)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]

        # statistics
        sta[phase]['epoch_loss'].append(epoch_loss)
        
        if phase == 'dev':
            print(epoch_loss, time.strftime("%H:%M:%S", time.localtime()))
        if phase == 'dev' and epoch in [0, 2, 4, 6, 9, 29, 49]:
            para_dict[epoch] = copy.deepcopy(net.state_dict())
        if phase == 'dev' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_wts = copy.deepcopy(net.state_dict())


net.load_state_dict(best_wts)

fig = plt.figure(figsize=(18, 4))

epochs = len(sta['train']['epoch_loss'])
min_d_loss = min(sta['dev']['epoch_loss'])
min_d_loss_index = sta['dev']['epoch_loss'].index(min_d_loss) + 1

# plot loss
t_loss = sta['train']['epoch_loss']
d_loss = sta['dev']['epoch_loss']
ax = plt.subplot(122)
tlline, = plt.plot(np.append(np.roll(t_loss, 1), t_loss[epochs - 1]), color='g')
dlline, = plt.plot(np.append(np.roll(d_loss, 1), d_loss[epochs - 1]), linestyle=":", color='r')
plt.grid(color="k", linestyle=":")
plt.legend((tlline, dlline), ('train', 'dev'))
plt.ylabel('loss')
plt.xlabel('iterations')
plt.title('CAE')
ax.set_xlim(1, epochs)
plt.show()

print("min train loss: " + str(min(t_loss)))
print("min dev loss: " + str(min_d_loss) + " at epoch " + str(min_d_loss_index))
print("corresponding train loss: " + str(t_loss[min_d_loss_index - 1]))

torch.save(net.state_dict(), './net/CAE.pt')

inputs, classes = next(iter(dataloaders['train']))
inputs = inputs[0:3]
out1 = torchvision.utils.make_grid(inputs)
imshow(out1)

for i in para_dict.keys():
    net.load_state_dict(para_dict[i])
    print(sta['dev']['epoch_loss'][i])
    reconstructed = net(inputs.to(device))
    out = torchvision.utils.make_grid(reconstructed.cpu().detach())
    imshow(out)


net.name = 'CAE'
path_net = './net/%s.pt'%net.name
torch.save(net.state_dict(), path_net)
path_sta = './figures/%s_sta.pt'%net.name
torch.save(sta, path_sta)