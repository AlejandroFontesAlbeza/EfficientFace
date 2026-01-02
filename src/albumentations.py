import torch
import torchvision.transforms as transforms 
import numpy as np


class Add_Gaussian_Noise(object): #add noise

    def __init__(self,mean=0.0,std=1.0,p=0.25, device = "cpu"): #mean: valor medio de ruido gaussiano, std = desviacion del ruido(cuanto ruido se aplica), p = probabilidad de aplicar ruido a la imagen
        self.mean = mean
        self.std = std
        self.p = p
        self.device = device
        
        
    def __call__(self, tensor):

        if np.random.uniform() < self.p:
            return tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean
        else:
            return tensor
        

class Add_Gaussian_Blur(object): #add blur-desenfoque
    def __init__(self, kernel_size= (7,7), sigma=(0.01,1.5), p= 0.25): #kernel_size: pixel mask 7x7, sigma = intensity(can be range or number), p = probabilidad de aplicar ruido a la imagen
        self.transform = transforms.GaussianBlur(kernel_size=kernel_size,sigma=sigma) #pytorch function
        self.p = p

    def __call__(self, img):
        if np.random.uniform() < self.p:
            img = self.transform(img)
        return img
    

class Adjust_Brightness(object): #modify img brightness
    def __init__(self, brightness_range = (-0.2,0.2), p = 0.25): #brightness_range: range of brightness modification(can be number), p = probabilidad de aplicar ruido a la imagen
        self.brightness_range = brightness_range
        self.p = p

    def __call__(self, tensor):
                
        if np.random.uniform() < self.p:
            brightness_factor = np.random.uniform(*self.brightness_range) # uniform(min,max), *to pass de tuple as numbers

            brightness_tensor = tensor + brightness_factor 

            return brightness_tensor
        else:
            return tensor




class Adjust_Contrast(object): #modify img contrast
    def __init__(self, contrast_range=(0.5,1.5),p=0.25): #contrast_range: range of contrast modification(can be number), p = probabilidad de aplicar ruido a la imagen
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, tensor):
        if np.random.uniform() < self.p:

            contrast_factor = np.random.uniform(*self.contrast_range)

            mean = tensor.mean()

            contrast_tensor = (tensor - mean ) * contrast_factor + mean 

            return contrast_tensor
        else:
            return tensor



class Invert_GrayScale(object): #invert grayscale
    def __init__(self, p= 0.25): #p = probabilidad de aplicar ruido a la imagen
        self.p = p
    def __call__(self, tensor):
        if np.random.uniform() < self.p:
            inverted_tensor = -1 *(tensor -0.5) + 0.5

            return inverted_tensor
        else:
            return tensor   