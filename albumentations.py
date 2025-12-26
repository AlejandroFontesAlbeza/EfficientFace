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
        


def transform_label_rotate(label, angles, img_size):
    # Apply rotation to label
    newpred = np.zeros(label.shape)
    rads = np.radians(angles)
    newpred[0::2] = (label[0::2] - img_size/2)*np.cos(rads) + (label[1::2]- img_size/2)*np.sin(rads) + img_size/2
    newpred[1::2] = - (label[0::2] - img_size/2)*np.sin(rads) + (label[1::2]- img_size/2)*np.cos(rads) + img_size/2
    return newpred


def transform_label_shift(label, shift_x, shift_y, img_size):
    # Apply shift to label
    newpred = np.zeros(label.shape)
    newpred[0::2] = label[0::2] + shift_x
    newpred[1::2] = label[1::2] + shift_y
    return newpred

def transform_label_scale(label, factor, img_size):
    # Apply scale to label
    newpred = np.zeros(label.shape)
    newpred[0::2] = (label[0::2] - img_size/2) * factor + img_size/2
    newpred[1::2] = (label[1::2] - img_size/2) * factor + img_size/2
    return newpred


class CombinedTransform(object):
    def __init__(self, img_size):
        # Initialize augmentation params:
        self.range_angle = 180
        self.img_size = img_size
        self.shift_pixel = self.img_size // 5
        self.scale_min = 0.6
        self.scale_max = 1.5
        self.p_aff = 0.3
        self.p_rot = 0.3

    def __call__(self, image, keypoints):
        
        # keypoints[np.isnan(keypoints)] = -1

        # Apply random rotation if probability condition is met
        if np.random.uniform() < self.p_rot:
            angles = np.random.uniform(-self.range_angle/2, self.range_angle/2)
        else:
            angles = 0
        
        # Apply scale rotation if probability condition is met
        if np.random.uniform() < self.p_aff:
            factors = np.random.uniform(self.scale_max, self.scale_min)
        else:
            factors = 1
        
        # Apply random shift if probability condition is met
        if np.random.uniform() < self.p_aff:
            shifts_x = np.random.uniform(-self.shift_pixel, self.shift_pixel)
            shifts_y = np.random.uniform(-self.shift_pixel, self.shift_pixel)
        else:
            shifts_x = 0
            shifts_y = 0

        # Apply affine transformation to the image
        image = transforms.functional.affine(
            image,
            angle=-angles,
            scale=factors,
            translate=(shifts_x, shifts_y),
            shear=0)
        
        # Apply corresponding transformation to keypoints
        keypoints = transform_label_rotate(keypoints, angles, self.img_size)
        keypoints = transform_label_scale(keypoints, factors, self.img_size)
        keypoints = transform_label_shift(keypoints, shifts_x, shifts_y, self.img_size)        

        return image, keypoints