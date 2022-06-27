import os
import sys

import cv2
import torch
from torchvision.transforms import ToTensor, ToPILImage

from train import OutpaintingGAN


def generate(
        image_path, model_path='logs/final/final.tar', bordered: bool = True, expand_size: int = 32):
    
    gpu_device = torch.device('cuda:0')
    cpu_device = torch.device("cpu")

    os.chdir('..')

    input_img = cv2.imread(image_path)
    if bordered:
        output_size = input_img.shape[1]
        cropped_size = output_size - expand_size * 2
    else:
        cropped_size = input_img.shape[1]
        output_size = cropped_size + expand_size * 2

    generator = OutpaintingGAN(learning_rate=0,
                               betas=(0,0),
                               cropd_size=cropped_size,
                               outpt_size=output_size,
                               loss_weights={}).generator()
    generator.to(gpu_device)
    
    checkpoint = torch.load(os.path.join(os.getcwd(), model_path))
    generator.load_state_dict(checkpoint['gen_model_state_dict'])

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
    gen_img = torch.unsqueeze(gen_img, 0).to(gpu_device)
    gen_img = generator(gen_img).to(cpu_device)
    gen_img = torch.squeeze(gen_img, 0)
    gen_img = ToPILImage()(gen_img)
    
    return gen_img


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    expand_size = int(sys.argv[3])
    os.chdir('..')
    
    generated_image = generate(input_path, bordered=False, expand_size=expand_size)
    generated_image.save(output_path)
