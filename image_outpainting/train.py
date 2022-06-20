import glob
import os
import random
import sys

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from pytorch_msssim import SSIM
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from torchvision.models import vgg19

from dataloader import OutpaintingDataset, mirror_image


l1_module = nn.L1Loss()
ssim_module_1 = SSIM(data_range=1, size_average=True, channel=1)
ssim_module_3 = SSIM(data_range=1, size_average=True, channel=3)
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
        self.conv6 = nn.Conv2d(64, out_channel, kernel_size=3, stride=1)  # 128 -> 128

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
    def __init__(self, cropped_size, output_size, in_channel):
        super().__init__()

        self.cropped_size = cropped_size
        self.output_size = output_size
        self.expand_size = (output_size - cropped_size) // 2

        self.lrelu = nn.LeakyReLU(0.2)
        self.inorm = nn.InstanceNorm2d(64)

        self.conv1 = nn.Conv2d(in_channel, 32, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, stride=2, padding=2)
        self.fc1 = nn.Linear(64 * (self.expand_size // 8) * (self.output_size // 16), 512)
        self.fc2 = nn.Linear(64 * (self.output_size // 32) ** 2, 512)
        self.fc3 = nn.Linear(1024, 1)

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
    def __init__(self, learning_rate, betas, cropped_size, output_size, loss_weights):
        super().__init__()

        self.cropped_size = cropped_size
        self.output_size = output_size
        self.expand_size = (output_size - cropped_size) // 2

        self.edge_generator = Generator(cropped_size, output_size, 1, 1)
        self.edge_discriminator = Discriminator(cropped_size, output_size, 1)

        self.generator = Generator(cropped_size, output_size, 4, 3)
        self.discriminator = Discriminator(cropped_size, output_size, 3)

        self.genedge_optimizer = optim.Adam(self.edge_generator.parameters(), lr=learning_rate, betas=betas)
        self.disedge_optimizer = optim.Adam(self.edge_discriminator.parameters(), lr=learning_rate, betas=betas)
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=betas)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=betas)
        self.loss_weights = loss_weights

    def train_step(self, inputs, gt, edge: bool):
        gt_left_crop = crop(gt, 0, 0, self.output_size, self.cropped_size // 2)
        gt_right_crop = crop(gt, 0, self.output_size - self.cropped_size // 2, self.output_size, self.cropped_size // 2)
        gt_cr = torch.cat((gt_right_crop, gt_left_crop), 3)
        if edge:
            gen = self.edge_generator
            dis = self.edge_discriminator
            gen_opt = self.genedge_optimizer
            dis_opt = self.disedge_optimizer
        else:
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

        if edge:
            loss_per = 1 - ssim_module_1(outs_cr, gt_cr)
        else:
            loss_per = 1 - ssim_module_3(outs_cr, gt_cr)

        outs_cr = outs_cr.expand(outs_cr.shape[0], 3, outs_cr.shape[2], outs_cr.shape[3])
        gt_cr = gt_cr.expand(gt_cr.shape[0], 3, gt_cr.shape[2], gt_cr.shape[3])
        loss_style = MSE_module(gram(vgg_module(outs_cr)), gram(vgg_module(gt_cr)))

        loss_adv = bce_module(dis(outputs), valid)

        loss_gen = self.loss_weights['pixel'] * loss_pxl \
                   + self.loss_weights['per'] * loss_per \
                   + self.loss_weights['style'] * loss_style \
                   + self.loss_weights['adv'] * loss_adv

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
        self.edge_generator.load_state_dict(checkpoint['edge_gen_model_state_dict'])
        self.discriminator.load_state_dict(checkpoint['dis_model_state_dict'])
        self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
        self.edge_discriminator.load_state_dict(checkpoint['edge_dis_model_state_dict'])
        self.disedge_optimizer.load_state_dict(checkpoint['edge_dis_optimizer_state_dict'])
        loss_dict = checkpoint['loss_dict']

        return epoch, loss_dict

    def show(self, image_name, edges_name, gt_name):
        # read pillow images
        cropped_image = Image.open(image_name)  # this is fot plot
        input_edges = Image.open(edges_name)
        gt_img = Image.open(gt_name)  # this is fot plot

        # mirror images
        input_img = mirror_image(cropped_image)
        input_edges = mirror_image(input_edges, mode="L")

        # transform to tensor
        input_img = torch.unsqueeze(transforms.ToTensor()(input_img).to(gpu_device), 0)
        input_edges = torch.unsqueeze(transforms.ToTensor()(input_edges).to(gpu_device), 0)

        # prepare to generator
        left_part = crop(input_img, 0, 0, self.output_size, self.cropped_size // 2)
        edges_pred = self.edge_generator(input_edges)
        inner_part = crop(edges_pred, 0, self.cropped_size // 2, self.output_size, self.expand_size * 2)
        inner_part = inner_part.expand(inner_part.shape[0], 3, inner_part.shape[2], inner_part.shape[3])
        right_part = crop(input_img,
                          0, self.output_size - self.cropped_size // 2, self.output_size, self.cropped_size // 2)
        hybrid_img = torch.cat((left_part, inner_part, right_part), 3)
        hybrid_img = torch.squeeze(hybrid_img, 0)
        hybrid_img = transforms.ToPILImage()(hybrid_img.to(cpu_device))
        hybrid_img = mirror_image(hybrid_img)  # this is for plot

        # predict
        pred_img = torch.squeeze(self.generator(torch.cat((input_img, edges_pred), 1)), 0)
        pred_img = transforms.ToPILImage()(pred_img.to(cpu_device))
        pred_img = mirror_image(pred_img)  # this is for plot


        fig, ax = plt.subplots(1, 4, figsize=(24, 6))
        ax[0].imshow(cropped_image)
        ax[0].set_title('Cropped Image'), ax[0].set_xticks([]), ax[0].set_yticks([])
        ax[1].imshow(hybrid_img)
        ax[1].set_title('Hybrid Image'), ax[1].set_xticks([]), ax[1].set_yticks([])
        ax[2].imshow(pred_img)
        ax[2].set_title('Predict Image'), ax[2].set_xticks([]), ax[2].set_yticks([])
        ax[3].imshow(gt_img)
        ax[3].set_title('GT'), ax[3].set_xticks([]), ax[3].set_yticks([])
        fig_name = model_name + '-epoch-' + str(epoch).zfill(3) + '.jpg'
        fig.savefig(os.path.join(os.getcwd(), 'logs', 'imgs', fig_name), dpi=50)


if __name__ == "__main__":

    os.chdir('..')
    cwd = os.getcwd()

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
    betas = tuple(map(float, config['optimizer']['adam_betas'].split(',')))
    loss_weights = config['loss']['loss_weights']
    model_name = str(config['model_name'])

    outpainting_gan = OutpaintingGAN(learning_rate, betas, cropped_size, output_size, loss_weights)
    outpainting_gan.to(gpu_device)
    epoch = 1
    loss_dict = {'edge_pxl': [], 'edge_per': [], 'edge_style': [], 'edge_adv': [], 'edge_dis': [],
                 'img_pxl': [], 'img_per': [], 'img_style': [], 'img_adv': [], 'img_dis': []}
    if config['mode'] == 'load':
        try:
            model_file = str(config['model_file'])
            model_path = os.path.join(os.getcwd(), 'logs', 'models', model_file)
            epoch, loss_dict = outpainting_gan.load_model(model_path)
        except:
            print("Failed to load model")

    num_workers = int(config['dataloader']['num_workers'])
    batch_size = int(config['dataloader']['batch_size'])

    trainset = OutpaintingDataset(os.path.join(os.getcwd(), 'dataset', 'train'))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    log_freq = int(config['train']['log_freq'])
    max_epoch = int(config['train']['max_epoch'])
    while epoch <= max_epoch:
        run_loss = [0] * 10
        for i, data in enumerate(trainloader, 0):
            # dataset input
            gt_imgs, crop_imgs, gt_edges, crop_edges = data
            gt_imgs = gt_imgs.to(gpu_device)
            crop_imgs = crop_imgs.to(gpu_device)
            gt_edges = gt_edges.to(gpu_device)
            crop_edges = crop_edges.to(gpu_device)

            # train edges model
            edge_run_losses = outpainting_gan.train_step(crop_edges, gt_edges, edge=True)

            # build hybrid image # TODO make function
            edges_pred = outpainting_gan.edge_generator(crop_edges)
            edged_crop_imgs = torch.cat((crop_imgs, edges_pred), 1)

            #left_part = crop(crop_imgs, 0, 0, output_size, cropped_size // 2)
            #edges_pred = outpainting_gan.edge_generator(crop_edges)
            #inner_part = crop(edges_pred, 0, cropped_size // 2, output_size, expand_size * 2)
            #inner_part = inner_part.expand(inner_part.shape[0], 3, inner_part.shape[2], inner_part.shape[3])
            #right_part = crop(crop_imgs, 0, output_size - cropped_size // 2, output_size, cropped_size // 2)
            #edged_crop_imgs = torch.cat((left_part, inner_part, right_part), 3)

            # train imgs model
            img_run_losses = outpainting_gan.train_step(edged_crop_imgs, gt_imgs, edge=False)

            run_loss = list(map(lambda x, y: x + y, run_loss, edge_run_losses + img_run_losses))
            if i % log_freq == log_freq - 1:
                print(
                    f'[{epoch}, {i + 1:5d}] '
                    f'loss_pxl: {run_loss[5] / log_freq:.3f} '
                    f'loss_per: {run_loss[6] / log_freq:.3f} '
                    f'loss_style: {run_loss[7] / log_freq:.10f} '
                    f'loss_adv: {run_loss[8] / log_freq:.3f} '
                    f'loss_dis: {run_loss[9] / log_freq:.3f} '
                )
                loss_dict['edge_pxl'].append(run_loss[0])
                loss_dict['edge_per'].append(run_loss[1])
                loss_dict['edge_style'].append(run_loss[2])
                loss_dict['edge_adv'].append(run_loss[3])
                loss_dict['edge_dis'].append(run_loss[4])
                loss_dict['img_pxl'].append(run_loss[5])
                loss_dict['img_per'].append(run_loss[6])
                loss_dict['img_style'].append(run_loss[7])
                loss_dict['img_adv'].append(run_loss[8])
                loss_dict['img_dis'].append(run_loss[9])
                run_loss = [0] * 10

        # show
        input_img_name = random.choice(glob.glob(os.path.join(cwd, 'dataset', 'train', 'cropped', '*.jpg')))
        edges_img_name = os.path.join(cwd, 'dataset', 'train', 'cr_edge', input_img_name.split('/')[-1])
        gt_img_name = os.path.join(cwd, 'dataset', 'train', 'gt', input_img_name.split('/')[-1])
        outpainting_gan.show(input_img_name, edges_img_name, gt_img_name)

        epoch += 1

    torch.save(
        {
            'epoch': epoch,
            'gen_model_state_dict': outpainting_gan.generator.state_dict(),
            'gen_optimizer_state_dict': outpainting_gan.gen_optimizer.state_dict(),
            'edge_gen_model_state_dict': outpainting_gan.edge_generator.state_dict(),
            'edge_gen_optimizer_state_dict': outpainting_gan.genedge_optimizer.state_dict(),
            'dis_model_state_dict': outpainting_gan.discriminator.state_dict(),
            'dis_optimizer_state_dict': outpainting_gan.dis_optimizer.state_dict(),
            'edge_dis_model_state_dict': outpainting_gan.edge_discriminator.state_dict(),
            'edge_dis_optimizer_state_dict': outpainting_gan.disedge_optimizer.state_dict(),
            'loss_dict': loss_dict
        },
        os.path.join(os.getcwd(), 'logs', 'models', model_name + '-epoch-' + str(epoch-1).zfill(3) + '.tar'))
