import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.models as models
from albumentations import CombinedTransform

### FUNCTIONS ###

def get_img(train,img_size):

    img = []
    print(f"Reading total images of dataset:{len(train)}")

    for i in range(len(train)):
        face_pixel = np.array(train['Image'].iloc[i].split(' '), dtype = 'float')

        face_pixel = np.reshape(face_pixel,(img_size,img_size,1))
        face_pixel /= 255

        img.append(face_pixel)
    img = np.array(img)

    return img


def show_image(img_size, img, keypoints_true, keypoints_pred=None, save_path = ''):

    #create figure, axis and we plot the image
    fig, ax = plt.subplots()
    ax.imshow(img,cmap='gray')

    #extract kpts annotated and we plot them

    ax.scatter(keypoints_true[0::2], keypoints_true[1::2],c='g')

    #if there are kpts predicted(later on the training) we plot them too

    if keypoints_pred is not None:
        ax.scatter(keypoints_pred[0::2], keypoints_pred[1::2],c='r')


    #adjust axis with img size and 1- cause origin is inverted

    ax.set_xlim(0,img_size)
    ax.set_ylim(img_size,0)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # cierra la figura para liberar memoria


def repeat_channels(x):
    return x.repeat(3,1,1)

def to_float(x,device):
    return x.float().to(device)


def train_one_epoch(train_loader, test_loader, model, criterion, optimizer, scheduler):
    
    # Calculate validation loss
    num_examples = 0
    val_losses = []
    model.eval()
    with torch.no_grad():                
        for val_images, val_labels in test_loader:

            val_preds = model(val_images)
            val_loss = criterion(val_preds[val_labels != -1], val_labels[val_labels != -1])
            num_examples += val_labels[val_labels != -1].shape[0]
            val_losses.append(val_loss.item())


    model.train()
    
    val_losses_epoch = (sum(val_losses) / num_examples) ** 0.5 
    
    
    
    # Train the model on the training set
    
    num_examples = 0
    losses = []
    for images, labels in tqdm(train_loader):
        optimizer.zero_grad()
        preds = model(images)
        preds[labels == -1] = -1

        loss = criterion(preds[labels != -1], labels[labels != -1])

        num_examples += torch.numel(preds[labels != -1])
        losses.append(loss.item())

        loss.backward()
        optimizer.step()


    losses_epoch = (sum(losses) / num_examples) ** 0.5   
    
    print(f"Loss on train: {losses_epoch}")
    print(f"Loss on val: {val_losses_epoch}")      

    # Adjust learning rate using scheduler based on validation loss
    scheduler.step(val_losses_epoch)

    # Adjust weight decay based on learning rate
    lr=optimizer.param_groups[0]['lr']
    print("Current Lr:", lr)

    return 0







### CLASSES ###

class FK_dataset_train(Dataset):
    def __init__(self, X, y, device, img_size, transform = None, transform_combined = None):
        self.X = X
        self.y = y
        self.device = device
        self.img_size = img_size
        self.transform = transform
        self.transform_combined = transform_combined


    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, index):
        image = self.X[index]
        kpts = self.y[index]
        kpts_combined = kpts.copy()


        if self.transform:
            image = self.transform(image)
        if self.transform_combined:
            combined_transform = CombinedTransform(img_size=self.img_size)
            image, kpts_combined = combined_transform(image, kpts_combined)
        


        kpts_combined[kpts_combined < 0] = -1
        kpts_combined[kpts_combined > self.img_size] = -1
        kpts_combined[np.isnan(kpts_combined)] = -1


        kpts_combined = torch.from_numpy(kpts_combined).to(self.device).float()

        sample = (image, kpts_combined)

        return sample
    
class FK_dataset_val(Dataset):
    def __init__(self, X, y, device, img_size, transform = None):
        self.X = X
        self.y = y
        self.device = device
        self.img_size = img_size
        self.transform = transform


    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, index):
        image = self.X[index]
        kpts = self.y[index]


        if self.transform:
            image = self.transform(image)

        
        kpts[kpts < 0] = -1
        kpts[kpts > self.img_size] = -1
        kpts[np.isnan(kpts)] = -1


        kpts = torch.from_numpy(kpts).to(self.device).float()

        sample = (image, kpts)

        return sample
    



class MyModel(nn.Module):

    def __init__(self, num_kpts = 30, grad_from = 2):
        super(MyModel,self).__init__()
        self.effnet = models.efficientnet_b0(pretrained = True)
        self.outputs_last_layer = 1280*3*3


        for name, param in self.effnet.features.named_parameters():
            if int(name.split('.')[0]) < grad_from:
                param.requires_grad = False


        self.regressor = nn.Sequential(nn.Dropout(p=0.6),
                                       nn.Linear(self.outputs_last_layer, num_kpts)
                                       )


    def forward(self,x):
        x = self.effnet.features(x)
        x = torch.flatten(x,1)
        x = self.regressor(x)

        return x
