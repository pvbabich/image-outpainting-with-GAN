import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import crop
import os
import cv2
import random
import scipy
import skimage
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.autograd import Variable
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # downsampling block
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2) #128 -> 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) #128 -> 64
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #64 -> 64
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) #64 -> 32
        self.convd1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2) #32 -> 32
        self.convd2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4) #32 -> 32
        self.convd3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8) #32 -> 32
        
        # upsampling block
        self.convt1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1) #32 -> 64
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1) #64 -> 64
        self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1) #64 -> 128
        self.conv6 = nn.Conv2d(64, 3, kernel_size=3, stride=1) #128 -> 128
        
        self.norm64 = nn.BatchNorm2d(64, eps=0.8)
        self.norm128 = nn.BatchNorm2d(128, eps=0.8)
        self.norm256 = nn.BatchNorm2d(256, eps=0.8)
        


    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.norm128(self.conv2(x)))
        x = F.relu(self.norm128(self.conv3(x)))
        x = F.relu(self.norm256(self.conv4(x)))
        x = F.relu(self.norm256(self.convd1(x)))
        x = F.relu(self.norm256(self.convd2(x)))
        x = F.relu(self.norm256(self.convd3(x)))
        
        x = F.relu(self.norm128(self.convt1(x)))
        x = F.relu(self.norm128(self.conv5(x)))
        x = F.relu(self.norm64(self.convt2(x)))
        x = torch.tanh(self.conv6(x))

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cropped_size = 64
        self.output_size = 128
        self.expand_size = 32
        
        
        self.lrelu = nn.LeakyReLU(0.2)
        self.inorm = nn.InstanceNorm2d(64)
        
        self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, stride=2, padding=2)
        self.fc1 = nn.Linear(self.expand_size**2, 512)
        self.fc2 = nn.Linear((self.cropped_size//2)**2, 512)
        self.fc3 = nn.Linear(1536, 1)

    def forward(self, x):
        
        y_left = crop(x, 0, 0, self.output_size, self.expand_size)
        y_right = crop(x, 0, self.output_size-self.expand_size, self.output_size, self.expand_size)
        
        y_left = self.lrelu(self.conv1(y_left))
        y_left = self.lrelu(self.inorm(self.conv2(y_left)))
        y_left = self.lrelu(self.inorm(self.conv3(y_left)))
        y_left = self.lrelu(self.inorm(self.conv3(y_left)))
        y_left = torch.flatten(y_left, start_dim=1)
        y_left = self.fc1(y_left)
        
        y_right = self.lrelu(self.conv1(y_right))
        y_right = self.lrelu(self.inorm(self.conv2(y_right)))
        y_right = self.lrelu(self.inorm(self.conv3(y_right)))
        y_right = self.lrelu(self.inorm(self.conv3(y_right)))
        y_right = torch.flatten(y_right, start_dim=1)
        y_right = self.fc1(y_right)
        
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.inorm(self.conv2(x)))
        x = self.lrelu(self.inorm(self.conv3(x)))
        x = self.lrelu(self.inorm(self.conv3(x)))
        x = self.lrelu(self.inorm(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc2(x)
        
        x = torch.cat((y_left, x, y_right), 1)
        x = nn.Sigmoid()(self.fc3(x))
        
        return x
    
class OutpaintingDataset(Dataset):
    def __init__(self,folder_path):
        super(OutpaintingDataset).__init__()
        # Get image list
        self.cropped_list = glob.glob(os.path.join(folder_path,'cropped')+'/*')
        self.gt_list = glob.glob(os.path.join(folder_path,'gt')+'/*')
        # Calculate len
        self.data_len = len(self.gt_list)
        # Transforms
        self.to_tensor = transforms.ToTensor()
        
    def __getitem__(self, index):
        cropped_image_path = self.cropped_list[index]
        gt_image_path = self.gt_list[index]
        # Open image
        cr_as_im = Image.open(cropped_image_path)
        gt_as_im = Image.open(gt_image_path)
        # Transform image to tensor
        cr_as_tensor = self.to_tensor(cr_as_im)
        gt_as_tensor = self.to_tensor(gt_as_im)
        
        return cr_as_tensor, gt_as_tensor

    def __len__(self):
        return self.data_len # of how many examples(images?) you have