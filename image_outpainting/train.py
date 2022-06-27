import glob
import os
import random
import sys

import torch
import yaml
from torch.utils.data import Dataset

from dataloader import OutpaintingDataset
from model import OutpaintingGAN

if __name__ == "__main__":

    os.chdir('..')
    cwd = os.getcwd()

    gpu_device = torch.device('cuda:0')

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
    os.makedirs(os.path.join(cwd, 'logs', 'imgs', model_name), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'logs', 'models', model_name), exist_ok=True)

    outpainting_gan = OutpaintingGAN(learning_rate, betas, cropped_size, output_size, loss_weights)
    outpainting_gan.to(gpu_device)
    epoch = 1
    loss_dict = {'img_pxl': [], 'img_per': [], 'img_style': [], 'img_adv': [], 'img_dis': []}
    if config['mode'] == 'load':
        try:
            model_file = str(config['model_file'])
            model_path = os.path.join(os.getcwd(), 'logs', 'models', model_file)
            epoch, loss_dict = outpainting_gan.load_model(model_path)
            print("Loaded ", model_path)
            epoch += 1
        except:
            print("Failed to load model")

    num_workers = int(config['dataloader']['num_workers'])
    batch_size = int(config['dataloader']['batch_size'])

    trainset = OutpaintingDataset(os.path.join(os.getcwd(), 'dataset', 'train'))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    log_freq = int(config['train']['log_freq'])
    max_epoch = int(config['train']['max_epoch'])
    while epoch <= max_epoch:
        run_loss = [0] * 9
        for i, data in enumerate(trainloader, 0):
            # dataset input
            gt_imgs, crop_imgs = data
            gt_imgs = gt_imgs.to(gpu_device)
            crop_imgs = crop_imgs.to(gpu_device)

            # train model
            img_run_losses = outpainting_gan.train_step(crop_imgs, gt_imgs)

            run_loss = list(map(lambda x, y: x + y, run_loss, img_run_losses))
            if i % log_freq == log_freq - 1:
                print(
                    f'[{epoch}, {i + 1:5d}] '
                    f'loss_pxl: {run_loss[0] / log_freq:.4f} '
                    f'loss_per: {run_loss[1] / log_freq:.4f} '
                    f'loss_style (1e+5): {run_loss[2] * 1e+6 / log_freq:.4f} '
                    f'loss_adv: {run_loss[3] / log_freq:.4f} '
                    f'loss_dis: {run_loss[4] / log_freq:.4f} '
                )
                loss_dict['img_pxl'].append(run_loss[0])
                loss_dict['img_per'].append(run_loss[1])
                loss_dict['img_style'].append(run_loss[2])
                loss_dict['img_adv'].append(run_loss[3])
                loss_dict['img_dis'].append(run_loss[4])
                run_loss = [0] * 5

                # show
                input_img_path = random.choice(glob.glob(os.path.join(cwd, 'dataset', 'train', 'cropped', '*.jpg')))
                gt_img_path = os.path.join(cwd, 'dataset', 'train', 'gt', input_img_path.split('/')[-1])
                fig_name = model_name + '-epoch-' + str(epoch).zfill(3) + '-' + str(i + 1) + '.jpg'
                fig_path = os.path.join(cwd, 'logs', 'imgs', model_name, fig_name)
                outpainting_gan.show(input_img_path, gt_img_path, fig_path)

        # save model
        if epoch % 5 == 0:
            save_path = os.path.join(os.getcwd(), 'logs', 'models', model_name,
                             model_name + '-epoch-' + str(epoch).zfill(3) + '.tar')
            torch.save(
                {
                    'epoch': epoch,
                    'gen_model_state_dict': outpainting_gan.generator.state_dict(),
                    'gen_optimizer_state_dict': outpainting_gan.gen_optimizer.state_dict(),
                    'dis_model_state_dict': outpainting_gan.discriminator.state_dict(),
                    'dis_optimizer_state_dict': outpainting_gan.dis_optimizer.state_dict(),
                    'loss_dict': loss_dict
                },
                save_path)
            print('Saved', save_path)

        epoch += 1
