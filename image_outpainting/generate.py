import os
import sys

import cv2
import torch
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms.functional import crop

from train import OutpaintingGAN


def generate(
        image_path, model_path='logs_model/final/final.tar', expand_size: int = 32):
    
    gpu_device = torch.device('cuda:0')
    cpu_device = torch.device("cpu")

    os.chdir('..')

    print("Open input mage:", image_path)
    input_img = cv2.imread(image_path)
    cropped_size = input_img.shape[1]
    output_size = cropped_size + expand_size * 2

    generator = OutpaintingGAN(lr=0.0004,
                               dis_lr=0.0004,
                               betas=(0.9, 0.999),
                               cropd_size=cropped_size,
                               outpt_size=output_size,
                               loss_weights={}).generator
    generator.to(gpu_device)
    
    checkpoint = torch.load(os.path.join(os.getcwd(), model_path))
    generator.load_state_dict(checkpoint['gen_model_state_dict'])

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
    gen_img = torch.cat((crop(gen_img, 0, output_size // 2, output_size, output_size // 2),
                         crop(gen_img, 0, 0, output_size, output_size // 2)), 3)
    gen_img = generator(gen_img).to(cpu_device)
    gen_img = torch.cat((crop(gen_img, 0, output_size // 2, output_size, output_size // 2),
                         crop(gen_img, 0, 0, output_size, output_size // 2)), 3)
    gen_img = torch.squeeze(gen_img, 0)
    gen_img = ToPILImage()(gen_img)
    
    return gen_img


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    expand_size = int(sys.argv[3])
    
    generated_image = generate(input_path, expand_size=expand_size)
    generated_image.save(output_path)
    print("Saved generated as", output_path)
