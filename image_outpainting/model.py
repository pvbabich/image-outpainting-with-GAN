import torch
import torch.optim as optim
from pytorch_msssim import SSIM
from torch import nn
from torch.nn.utils import spectral_norm
from torchvision.models import VGG19_Weights
from torchvision.models import vgg19
from torchvision.transforms.functional import crop

l1_module = nn.L1Loss()
ssim_module = SSIM(data_range=1, size_average=True, channel=3)
MSE_module = nn.MSELoss()
bce_module = nn.BCELoss()
gpu_device = torch.device('cuda:0')
cpu_device = torch.device("cpu")
vgg_module = vgg19(weights=VGG19_Weights.DEFAULT).features.to(gpu_device).eval()


def gram(x):
    """ Calculate gram function of matrix """
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

        # down sampling block of the generator network
        self.downsampling_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, 64, kernel_size=5, stride=1, padding=2)),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # dilated block for the generator network
        self.dilated_block = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # up sampling block for the generator network
        self.upsampling_block = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(64, out_channel, kernel_size=3, stride=1))
        )

    def forward(self, x):
        """ Forward propagation of the generator network"""
        x = self.downsampling_block(x)
        x = self.dilated_block(x)
        x = self.upsampling_block(x)
        x = nn.Sigmoid()(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, cropped_size, output_size, in_channel):
        super().__init__()
        self.cropped_size = cropped_size
        self.output_size = output_size
        self.expand_size = (output_size - cropped_size) // 2

        # local discriminator pays more attention to the reconstructed part of the image,
        # which improves the quality of the generated objects
        self.local_discriminator = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, 64, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(64, 128, 5, stride=2, padding=2)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(128, 128, 5, stride=2, padding=2)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(128, 128, 5, stride=2, padding=2)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        # global discriminator ensures training stability and draws attention to the entire image
        self.global_discriminator = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, 64, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(64, 128, 5, stride=2, padding=2)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(128, 128, 5, stride=2, padding=2)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(128, 128, 5, stride=2, padding=2)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(128, 128, 5, stride=2, padding=2)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        # spectral norm is used to ensure training stability
        self.fc1 = spectral_norm(nn.Linear(128 * (self.expand_size // 8) * (self.output_size // 16), 1024))
        self.fc2 = spectral_norm(nn.Linear(128 * (self.output_size // 32) ** 2, 1024))
        self.fc3 = spectral_norm(nn.Linear(2048, 1))

    def forward(self, x):
        # cut out the center of the image, this part will go to the input of the local discriminator
        y_in = crop(x, 0, self.cropped_size // 2, self.output_size, self.expand_size * 2)
        y_in = self.local_discriminator(y_in)
        y_in = torch.flatten(y_in, start_dim=1)
        y_in = self.fc1(y_in)

        # uncropped image is fed to the input of the global discriminator
        x = self.global_discriminator(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc2(x)

        # the concatenated outputs of the local and global discriminator are fed into the hidden layer
        # of the discriminator and then into the sigmoid function
        x = torch.cat((y_in, x), 1)
        x = self.fc3(x)
        x = nn.Sigmoid()(x)

        return x


class OutpaintingGAN(nn.Module):
    def __init__(self, lr, dis_lr, betas, cropd_size, outpt_size, loss_weights):
        super().__init__()
        self.cropped_size = cropd_size
        self.output_size = outpt_size
        self.expand_size = (self.output_size - self.cropped_size) // 2

        self.generator = Generator(self.cropped_size, self.output_size, 3, 3)
        self.discriminator = Discriminator(self.cropped_size, self.output_size, 3)

        # we use different learning rates for generator and discriminator for more customization
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=dis_lr, betas=betas)
        self.loss_weights = loss_weights

    def train_step(self, inputs, gt):
        # get known part of ground truth
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

        loss_style = MSE_module(gram(vgg_module(outputs)), gram(vgg_module(gt)))

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
