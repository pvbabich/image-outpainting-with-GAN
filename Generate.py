import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms.functional import crop
import os
import cv2
import random
import skimage
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import sys
from Model import Generator

def generate(image_path, generator_path='generator_final.tar', bordered=True, expand_size=None):
    
    gpu_device = torch.device('cuda:0')
    cpu_device = torch.device("cpu")
    
    gen = Generator()
    gen.to(gpu_device)
    
    checkpoint_G = torch.load(generator_path)
    gen.load_state_dict(checkpoint_G['model_state_dict'])
    
    input_img = cv2.imread(image_path)
    if not bordered:
        input_img = cv2.copyMakeBorder(
            input_img,
            top=0, 
            bottom=0, 
            left=expand_size, 
            right=expand_size, 
            borderType=cv2.BORDER_CONSTANT,  
            value=0
        )
    gen_img = ToTensor()(input_img)
    gen_img = torch.unsqueeze(gen_img,0).to(gpu_device)
    gen_img = gen(gen_img).to(cpu_device)
    gen_img = torch.squeeze(gen_img,0)
    gen_img = ToPILImage()(gen_img)
    
    return gen_img

if (__name__ == "__main__"):
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    expand_size = int(sys.argv[3])
    
    generated_image = generate(input_path, bordered=False, expand_size=expand_size)
    generated_image.save(output_path)