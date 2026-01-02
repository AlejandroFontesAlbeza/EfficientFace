### DATA MANIPULATION ###

import random ##allows generation of random numbers and selections

import numpy as np #support for numerical operations and arrays
import pandas as pd #data manipulation and analyis tools
from matplotlib import pyplot as plt #library for visualize data


#### PYTORCH ###

import torch #deep learning framework
import torch.nn as nn #provides various neural network layers and functions
import torch.optim as optim ##implementes optimization algorithms funcions
from torch.utils.data import DataLoader #functions from torch to create/load datasets

import torchvision.transforms as transforms #common image transformations functions

### MY_MODULES ###

from utils import get_img, repeat_channels, to_float, FK_dataset, MyModel, train_one_epoch
from albumentations import Invert_GrayScale, Add_Gaussian_Noise, Add_Gaussian_Blur, Adjust_Brightness, Adjust_Contrast



img_size = 96 #input image size(depends on dataset for pixel's array)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_samples = 10000




#loading de dataset

train = pd.read_csv("../dataset/training.zip", compression='zip',nrows=max_samples)


#extracting images for train and kpts por training dataset

img_train = get_img(train, img_size)
kpts_train = train.drop('Image', axis=1).values.astype('float')



###APPLY ALBUMENTATIONS, TOGETHER
        
#we only normalize and do basic operations for image val
transform_image_val = transforms.Compose([
    transforms.ToTensor(),
    repeat_channels,  
    transforms.Normalize(mean=[0.4897,0.4897,0.4897], std = [0.2330,0.2330,0.2330]), 
    lambda x: to_float(x, device=device) 
])



# #normalize and apply albumentations
transform_image_train = transforms.Compose([
  # Convert numpy image to tensor image
    transforms.ToTensor(),
    Invert_GrayScale(p=0.1), 
    repeat_channels,  
    lambda x: to_float(x, device=device),
    Add_Gaussian_Blur(kernel_size=(7, 7), sigma=(0.01, 1.5), p=0.25),  
    Add_Gaussian_Noise(mean=0., std=0.2, p=0.25, device=device),  
    transforms.Normalize(mean=[0.4897,0.4897,0.4897], std = [0.2330,0.2330,0.2330]),
    Adjust_Brightness(p=0.25), 
    Adjust_Contrast(p=0.25)
    ])

###CREATE DATASETS AND DATALOADERS

val_samples = random.sample(range(len(img_train)), len(img_train)//20)  ## 20% of the dataset we use it for val data
train_samples = np.setdiff1d(range(len(img_train)), val_samples) ## setdiff1d --------> from this array1 and array2 take the ones that are not used on array2


train_data = FK_dataset(img_train[train_samples],kpts_train[train_samples], device=device, img_size=img_size,transform=transform_image_train)

val_data = FK_dataset(img_train[val_samples],kpts_train[val_samples], device=device, img_size=img_size, transform=transform_image_val)


batch_size = 200

train_loader = DataLoader(train_data,batch_size,shuffle=True)
val_loader = DataLoader(val_data,batch_size,shuffle=True)







###MODEL DEFINITION ----> EFFICIENTnET-B0

##we create the model
model = MyModel().to(device)



####TRAINING

criterion = nn.MSELoss(reduction='sum')

lr = 0.001

optimizer = optim.Adam(model.parameters(), lr=lr)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.6, threshold=0.04, min_lr=1e-8)    


def main() -> None:

    epoch_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(100):
        print('Epoch:', epoch)
        epoch_list.append(epoch)
        train_loss,val_loss = train_one_epoch(train_loader, val_loader, model, criterion, optimizer, scheduler)
        print(' ')
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)


    torch.save(model.state_dict(),"eficientNetFace.pth")
    print("modelo guardado correctamente")

    plt.figure(figsize=(10,5))

    plt.plot(epoch_list,train_loss_list, label = "Training Loss")
    plt.plot(epoch_list,val_loss_list, label = "Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("../resources/dataLoss.png")
    plt.close()
   



if __name__ == "__main__":
    main()














    



        