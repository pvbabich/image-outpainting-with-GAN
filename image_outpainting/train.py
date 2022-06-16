import glob
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from pytorch_msssim import SSIM
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

from config.config import OutpaintingConfig
from dataloader import OutpaintingDataset


class Generator(nn.Module):
    def __init__(self, cropped_size, output_size):
        super().__init__()

        self.cropped_size = cropped_size
        self.output_size = output_size
        self.expand_size = (output_size - cropped_size) // 2

        # downsampling block
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)  # 128 -> 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 128 -> 64
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # 64 -> 64
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.convd1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)  # 32 -> 32
        self.convd2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)  # 32 -> 32
        self.convd3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)  # 32 -> 32

        # upsampling block
        self.convt1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1)  # 32 -> 64
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1)  # 64 -> 64
        self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1)  # 64 -> 128
        self.conv6 = nn.Conv2d(64, 3, kernel_size=3, stride=1)  # 128 -> 128

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
    def __init__(self, cropped_size, output_size):
        super().__init__()

        self.cropped_size = cropped_size
        self.output_size = output_size
        self.expand_size = (output_size - cropped_size) // 2

        self.lrelu = nn.LeakyReLU(0.2)
        self.inorm = nn.InstanceNorm2d(64)

        self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, stride=2, padding=2)
        self.fc1 = nn.Linear(self.expand_size ** 2, 512)
        self.fc2 = nn.Linear((self.cropped_size // 2) ** 2, 512)
        self.fc3 = nn.Linear(1536, 1)

    def forward(self, x):
        y_left = crop(x, 0, 0, self.output_size, self.expand_size)
        y_right = crop(x, 0, self.output_size - self.expand_size, self.output_size, self.expand_size)

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


class OutpaintingGAN(nn.Module):
    def __init__(self, learning_rate, betas, cropped_size, output_size, loss_weights):
        super().__init__()

        self.cropped_size = cropped_size
        self.output_size = output_size
        self.expand_size = (output_size - cropped_size) // 2

        self.generator = Generator(cropped_size, output_size)
        self.discriminator = Discriminator(cropped_size, output_size)

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=betas)
        self.dis_optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=betas)
        self.loss_weights = loss_weights

    def train_step(self, inputs, gt):
        gt_cr = crop(gt, 0, self.expand_size, self.output_size, self.cropped_size)
        self.gen_optimizer.zero_grad()
        outputs = self.generator(inputs)

        valid = torch.ones(outputs.shape[0], 1).to(gpu_device)
        fake = torch.zeros(outputs.shape[0], 1).to(gpu_device)

        l1_module = nn.L1Loss()
        loss_pxl = l1_module(crop(outputs, 0, expand_size, output_size, cropped_size), gt_cr)

        ssim_module = SSIM(data_range=1, size_average=True, channel=3)
        loss_per = 1 - ssim_module(crop(outputs, 0, expand_size, output_size, cropped_size), gt_cr)

        bce_module = nn.BCELoss()
        loss_adv = bce_module(self.discriminator(outputs), valid)

        loss_gen = self.loss_weights['pixel'] * loss_pxl \
                   + self.loss_weights['per'] * loss_per \
                   + self.loss_weights['adv'] * loss_adv

        loss_gen.backward()
        self.gen_optimizer.step()

        self.dis_optimizer.zero_grad()
        loss_valid = bce_module(self.discriminator(gt), valid)
        loss_fake = bce_module(self.discriminator(outputs.detach()), fake)
        loss_dis = loss_valid + loss_fake

        loss_dis.backward()
        self.dis_optimizer.step()

        return loss_pxl.item(), loss_per.item(), loss_adv.item(), loss_dis.item()

    def load_generator(self, generator_path):
        checkpoint_gen = torch.load(generator_path)
        self.generator.load_state_dict(checkpoint_gen['model_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint_gen['optimizer_state_dict'])
        epoch = checkpoint_gen['epoch']
        loss_pxl_array = checkpoint_gen['loss_pxl_array']
        loss_adv_array = checkpoint_gen['loss_adv_array']

        return epoch, loss_pxl_array, loss_adv_array

    def load_discriminator(self, discriminator_path):
        checkpoint_dis = torch.load(discriminator_path)
        self.discriminator.load_state_dict(checkpoint_dis['model_state_dict'])
        self.dis_optimizer.load_state_dict(checkpoint_dis['optimizer_state_dict'])
        loss_dis_array = checkpoint_dis['loss_D_array']

        return loss_dis_array


if __name__ == "__main__":

    os.chdir('..')
    gpu_device = torch.device('cuda:0')
    cpu_device = torch.device("cpu")

    if len(sys.argv) == 1:
        config_name = "default.yaml"
    else:
        config_name = sys.argv[1]
    with open(os.path.join(os.getcwd(), 'config', config_name), 'r') as stream:
        config = yaml.safe_load(stream)

    cropped_size = int(config['size']['cropped_size'])
    output_size = int(config['size']['output_size'])
    expand_size = (output_size - cropped_size) // 2
    learning_rate = float(config['optimizer']['learning_rate'])
    betas = tuple(map(float,config['optimizer']['adam_betas'].split(',')))
    loss_weights = config['loss']['loss_weights']
    model_name = str(config['model_name'])

    outpainting_gan = OutpaintingGAN(learning_rate, betas, cropped_size, output_size, loss_weights)
    outpainting_gan.to(gpu_device)
    epoch = 1
    loss_pxl_array = []
    loss_per_array = []
    loss_adv_array = []
    loss_dis_array = []
    if config['mode'] == 'load':
        try:
            generator_path = os.path.join(os.getcwd(), 'logs', 'models', 'gen-' + model_name + '.tar')
            discriminator_path = os.path.join(os.getcwd(), 'logs', 'models', 'dis-' + model_name + '.tar')
            epoch, loss_pxl_array, loss_adv_array = outpainting_gan.load_generator(generator_path)
            loss_dis_array = outpainting_gan.load_discriminator(generator_path)
        except:
            print("Failed to load model")

    num_workers = int(config['dataloader']['num_workers'])
    batch_size = int(config['dataloader']['batch_size'])

    trainset = OutpaintingDataset(os.path.join(os.getcwd(), 'dataset', 'train'))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    log_freq = int(config['train']['log_freq'])
    max_epoch = int(config['train']['max_epoch'])
    while epoch <= max_epoch:
        running_loss_pxl = 0.0
        running_loss_per = 0.0
        running_loss_adv = 0.0
        running_loss_dis = 0.0
        for i, data in enumerate(trainloader, 0):
            # dataset input
            inputs, gt = data
            inputs = inputs.to(gpu_device)
            gt = gt.to(gpu_device)
            running_losses = outpainting_gan.train_step(inputs, gt)

            running_loss_pxl += running_losses[0]
            running_loss_per += running_losses[1]
            running_loss_adv += running_losses[2]
            running_loss_dis += running_losses[3]

            if i % log_freq == log_freq - 1:
                print(
                    f'[{epoch}, {i + 1:5d}] '
                    f'loss_pxl: {running_loss_pxl / log_freq:.4f} '
                    f'loss_per: {running_loss_per / log_freq:.3f} '
                    f'loss_adv: {running_loss_adv / log_freq:.3f} '
                    f'loss_D: {running_loss_dis / log_freq:.3f}'
                )
                loss_pxl_array.append(running_loss_pxl / log_freq)
                loss_per_array.append(running_loss_per / log_freq)
                loss_adv_array.append(running_loss_adv / log_freq)
                loss_dis_array.append(running_loss_dis / log_freq)
                running_loss_pxl = 0.0
                running_loss_per = 0.0
                running_loss_adv = 0.0
                running_loss_dis = 0.0

        in_img = transforms.ToPILImage()(torch.squeeze(inputs[0], 0).to(cpu_device))
        gen_img = transforms.ToPILImage()(torch.squeeze(outpainting_gan.generator(inputs)[0], 0).to(cpu_device))
        gt_img = transforms.ToPILImage()(torch.squeeze(gt[0], 0).to(cpu_device))

        fig, axarr = plt.subplots(1, 3, figsize=(18, 6))
        axarr[0].imshow(in_img)
        axarr[1].imshow(gen_img)
        axarr[2].imshow(gt_img)
        fig_name = model_name + '-epoch-' + str(epoch).zfill(3) + '.jpg'
        fig.savefig(os.path.join(os.getcwd(), 'logs', 'imgs', fig_name), dpi=50)

        epoch += 1

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': outpainting_gan.generator.state_dict(),
            'optimizer_state_dict': outpainting_gan.gen_optimizer.state_dict(),
            'loss_pxl_array': loss_pxl_array,
            'loss_per_array': loss_per_array,
            'loss_adv_array': loss_adv_array,
        },
        os.path.join(os.getcwd(), 'logs', 'models', 'gen-' + model_name + '-epoch-' + str(epoch-1).zfill(3) + '.tar'))

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': outpainting_gan.discriminator.state_dict(),
            'optimizer_state_dict': outpainting_gan.dis_optimizer.state_dict(),
            'loss_dis_array': loss_dis_array,
        },
        os.path.join(os.getcwd(), 'logs', 'models', 'dis-' + model_name + '-epoch-' + str(epoch-1).zfill(3) + '.tar'))
