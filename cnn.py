import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv')
df1=pd.read_csv('valid.csv')

transforms=transforms.Compose([
       transforms.Grayscale(num_output_channels=1),
       transforms.ToTensor()
    ])

dataset_train=ImageFolder(root='processed_train/',transform=transforms)
dataset_valid=ImageFolder(root='processed_valid/',transform=transforms)

train_loader=DataLoader(dataset=dataset_train,batch_size=1000,shuffle=True)
val_loader=DataLoader(dataset=dataset_valid,batch_size=1000,shuffle=False)

class ConvolutionBlock1(nn.Module):

    def __init__(self,in_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,2*in_channels,3,1,padding=0)
        self.conv2=nn.Conv2d(2*in_channels,3*in_channels,3,1,padding=0)

    def forward(self,img):
        out=F.relu(self.conv1(img))
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2,2)

        return out


class ConvolutionBlock2(nn.Module):

    def __init__(self,in_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,2*in_channels,3,1,padding=0)
        self.conv2=nn.Conv2d(2*in_channels,3*in_channels,3,1,padding=0)
        self.conv3=nn.Conv2d(3*in_channels,3*in_channels,3,1,padding=0)

    def forward(self,img):
        out=F.relu(self.conv1(img))
        out=F.relu(self.conv2(out))
        out=F.relu(self.conv3(out))
        out=F.max_pool2d(out,2,2)

        return out

class Convoluter(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convblock1 = ConvolutionBlock1(in_channels)
        self.convblock2 = ConvolutionBlock1(self.convblock1.conv2.out_channels)
        self.convblock3 = ConvolutionBlock2(self.convblock2.conv2.out_channels)
        self.convblock4 = ConvolutionBlock2(self.convblock3.conv3.out_channels)
        self.convblock5 = ConvolutionBlock2(self.convblock4.conv3.out_channels)

    def forward(self, batch):
        out = F.relu(self.convblock1(batch))
        out = F.relu(self.convblock2(out))
        out = F.relu(self.convblock3(out))
        out = F.relu(self.convblock4(out))
        out = F.relu(self.convblock5(out))
        return out


class Classifier(nn.Module):

    def __init__(self,n):
        super().__init__()
        self.convoluter=Convoluter(1)
        self.dense1=nn.Linear(n,2*n//3)
        self.dropout1=nn.Dropout(0.2)
        self.dense2=nn.Linear(2*n//3,n//3)
        self.dropout2=nn.Dropout(0.1)
        self.dense3=nn.Linear(n//3,2)

    def forward(self,batch):
        batch=self.convoluter(batch)
        batch=torch.flatten(batch,start_dim=1)
        out=F.leaky_relu(self.dense1(batch))
        out=self.dropout1(out)
        out=F.leaky_relu(self.dense2(out))
        out=self.dropout2(out)
        out=self.dense3(out)

        return out

model=Classifier(972)

torch.manual_seed(123)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

epochs=30


train_losses=[]
val_losses=[]
model.train()
for i in range(epochs):
    train_loss=0
    for batch in train_loader:
        inputs,labels=batch[0],batch[1]
        logits=model(inputs)
        loss=criterion(logits,labels)
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss)

    model.eval()
    val_loss=0
    with torch.no_grad():
       for batch in val_loader:
          inputs,labels=batch[0],batch[1]
          logits=model(inputs)
          loss=criterion(logits,labels)
          val_loss+=loss.item()

    val_losses.append(val_loss)

    print(f"The train loss after epoch {i} is {train_loss} the val loss is {val_loss}")



y=np.arange(epochs)
plt.figure(figsize=(10,10))
plt.plot(y,np.array(train_losses),label='Train Loss')
plt.plot(y,np.array(val_losses),label="Validation Loss")
plt.legend(loc='Upper Left')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

torch.save(model.state_dict(),"model.pt")
