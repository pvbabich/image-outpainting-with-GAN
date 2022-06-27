import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from pytorch_msssim import SSIM
from torch import nn
from torch.nn.utils import spectral_norm
from torchvision import transforms
from torchvision.models import vgg19
from torchvision.transforms.functional import crop

from dataloader import mirror_image

l1_module = nn.L1Loss()
ssim_module = SSIM(data_range=1, size_average=True, channel=3)
MSE_module = nn.MSELoss()
bce_module = nn.BCELoss()
gpu_device = torch.device('cuda:0')
cpu_device = torch.device("cpu")
vgg_module = vgg19(pretrained=True).features.to(gpu_device).eval()


def gram(x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_t = f.transpose(1, 2)
    return f.bmm(f_t) / (h * w * ch)


class Generator(nn.Module):
    def __init__(self, cropped_size, output_size, in_channel, out_channel):
        super().__init__()

        self.cropped_size = cropped_size
        self.output_size = output_size
        self.expand_size = (output_size - cropped_size) // 2

        # downsampling block
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=5, stride=1, padding=2)  # 128 -> 128
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))  # 128 -> 64
        self.conv3 = spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))  # 64 -> 64
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))  # 64 -> 32
        self.convd1 = spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2))  # 32 -> 32
        self.convd2 = spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4))  # 32 -> 32
        self.convd3 = spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8))  # 32 -> 32

        # upsampling block
        self.convt1 = spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1))  # 32 -> 64
        self.conv5 = spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1))  # 64 -> 64
        self.convt2 = spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1))  # 64 -> 128
        self.conv6 = spectral_norm(nn.Conv2d(64, out_channel, kernel_size=3, stride=1))  # 128 -> 128

        self.norm64 = nn.BatchNorm2d(64)
        self.norm128 = nn.BatchNorm2d(128)
        self.norm256 = nn.BatchNorm2d(256)

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
    def __init__(self, cropped_size, output_size, in_channel):
        super().__init__()

        self.cropped_size = cropped_size
        self.output_size = output_size
        self.expand_size = (output_size - cropped_size) // 2

        self.lrelu = nn.LeakyReLU(0.2)

        self.conv1 = spectral_norm(nn.Conv2d(in_channel, 32, 5, stride=2, padding=2))
        self.conv2 = spectral_norm(nn.Conv2d(32, 64, 5, stride=2, padding=2))
        self.conv3 = spectral_norm(nn.Conv2d(64, 64, 5, stride=2, padding=2))
        self.fc1 = spectral_norm(nn.Linear(64 * (self.expand_size // 8) * (self.output_size // 16), 512))
        self.fc2 = spectral_norm(nn.Linear(64 * (self.output_size // 32) ** 2, 512))
        self.fc3 = spectral_norm(nn.Linear(1024, 1))

        self.inorm = nn.InstanceNorm2d(64)

    def forward(self, x):
        y_in = crop(x, 0, self.cropped_size // 2, self.output_size, self.expand_size * 2)

        y_in = self.lrelu(self.conv1(y_in))
        y_in = self.lrelu(self.inorm(self.conv2(y_in)))
        y_in = self.lrelu(self.inorm(self.conv3(y_in)))
        y_in = self.lrelu(self.inorm(self.conv3(y_in)))
        y_in = torch.flatten(y_in, start_dim=1)
        y_in = self.fc1(y_in)

        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.inorm(self.conv2(x)))
        x = self.lrelu(self.inorm(self.conv3(x)))
        x = self.lrelu(self.inorm(self.conv3(x)))
        x = self.lrelu(self.inorm(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc2(x)

        x = torch.cat((y_in, x), 1)
        x = nn.Sigmoid()(self.fc3(x))

        return x


class OutpaintingGAN(nn.Module):
    def __init__(self, learning_rate, betas, cropd_size, outpt_size, loss_weights):
        super().__init__()

        self.cropped_size = cropd_size
        self.output_size = outpt_size
        self.expand_size = (self.output_size - self.cropped_size) // 2

        self.generator = Generator(self.cropped_size, self.output_size, 3, 3)
        self.discriminator = Discriminator(self.cropped_size, self.output_size, 3)

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=betas, weight_decay=1e-5)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=betas, weight_decay=1e-5)
        self.loss_weights = loss_weights

    def train_step(self, inputs, gt):
        gt_left_crop = crop(gt, 0, 0, self.output_size, self.cropped_size // 2)
        gt_right_crop = crop(gt, 0, self.output_size - self.cropped_size // 2, self.output_size, self.cropped_size // 2)
        gt_cr = torch.cat((gt_right_crop, gt_left_crop), 3)

        gen = self.generator
        dis = self.discriminator
        gen_opt = self.gen_optimizer
        dis_opt = self.dis_optimizer

        outputs = gen(inputs)

        valid = torch.ones(outputs.shape[0], 1).to(gpu_device)
        fake = torch.zeros(outputs.shape[0], 1).to(gpu_device)

        gen_opt.zero_grad()

        outs_left_crop = crop(outputs, 0, 0, self.output_size, self.cropped_size // 2)
        outs_right_crop = crop(outputs,
                               0, self.output_size - self.cropped_size // 2, self.output_size, self.cropped_size // 2)
        outs_cr = torch.cat((outs_right_crop, outs_left_crop), 3)
        loss_pxl = l1_module(outs_cr, gt_cr)

        loss_per = 1 - ssim_module(outs_cr, gt_cr)

        loss_style = MSE_module(gram(vgg_module(outs_cr)), gram(vgg_module(gt_cr)))

        loss_adv = bce_module(dis(outputs), valid)

        loss_gen = self.loss_weights['pixel'] * loss_pxl + \
                   self.loss_weights['per'] * loss_per + \
                   self.loss_weights['style'] * loss_style + \
                   self.loss_weights['adv'] * loss_adv

        loss_gen.backward()
        gen_opt.step()

        dis_opt.zero_grad()
        loss_valid = bce_module(dis(gt), valid)
        loss_fake = bce_module(dis(outputs.detach()), fake)
        loss_dis = loss_valid + loss_fake

        loss_dis.backward()
        dis_opt.step()

        return loss_pxl.item(), loss_per.item(), loss_style.item(), loss_adv.item(), loss_dis.item()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch']
        self.generator.load_state_dict(checkpoint['gen_model_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.discriminator.load_state_dict(checkpoint['dis_model_state_dict'])
        self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
        loss_dict = checkpoint['loss_dict']

        return epoch, loss_dict

    def show(self, image_name, gt_name, fig_path):
        # read pillow images
        input_img = Image.open(image_name)  # this is for plot
        gt_img = Image.open(gt_name)  # this is for plot

        # mirror images
        cropped_image = mirror_image(input_img)

        # transform to tensor
        cropped_image = torch.unsqueeze(transforms.ToTensor()(cropped_image).to(gpu_device), 0)

        # predict
        pred_img = self.generator(cropped_image)

        # prepare to show
        pred_img = torch.squeeze(pred_img, 0)
        pred_img = transforms.ToPILImage()(pred_img.to(cpu_device))
        pred_img = mirror_image(pred_img, mode="RGB")  # this is for plot

        # show
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(input_img)
        ax[0].set_title('Cropped Image'), ax[0].set_xticks([]), ax[0].set_yticks([])
        ax[1].imshow(pred_img)
        ax[1].set_title('Predict Image'), ax[1].set_xticks([]), ax[1].set_yticks([])
        ax[2].imshow(gt_img)
        ax[2].set_title('Ground Truth'), ax[2].set_xticks([]), ax[2].set_yticks([])
        # fig_name = model_name + '-epoch-' + str(epoch).zfill(3) + '.jpg'
        fig.savefig(fig_path, dpi=50)
        plt.close()
