import sys

import cv2
import torch
from torchvision.transforms import ToTensor, ToPILImage

from image_outpainting.old.train_edged import Generator


def generate(
        image_path, model_path = 'logs/models/gen-final.tar', bordered: bool = True, expand_size: int = 32):
    
    gpu_device = torch.device('cuda:0')
    cpu_device = torch.device("cpu")

    input_img = cv2.imread(image_path)
    if bordered:
        output_size = input_img.shape[1]
        cropped_size = output_size - expand_size * 2
    else:
        cropped_size = input_img.shape[1]
        output_size = cropped_size + expand_size * 2

    generator = Generator(cropped_size, output_size)
    generator.to(gpu_device)
    
    checkpoint_gen = torch.load(generator_path)
    generator.load_state_dict(checkpoint_gen['model_state_dict'])

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
