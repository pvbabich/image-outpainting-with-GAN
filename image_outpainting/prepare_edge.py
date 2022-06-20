import glob
import os
import sys

import numpy as np
import tqdm
from PIL import Image, ImageOps
from skimage.feature import canny


def preprocessing(cropped_size, output_size, expand_size, target_dir='train'):

    os.chdir('..')
    cwd = os.getcwd()
    gt_names = glob.glob(os.path.join(cwd, 'dataset', target_dir, 'gt') + '/*.jpg')
    bar = tqdm.tqdm()
    for gt_img_path in gt_names:
        image_name = gt_img_path.split('/')[-1]

        gt_img = Image.open(gt_img_path)
        gt_img = gt_img.resize((output_size, output_size))
        gt_img.save(gt_img_path)

        cropped_img_path = os.path.join(cwd, 'dataset', target_dir, 'cropped', image_name)
        cropped_img = gt_img.crop((expand_size, 0, cropped_size + expand_size, output_size))
        cropped_img = ImageOps.expand(cropped_img, border=(expand_size, 0, expand_size, 0))
        cropped_img.save(cropped_img_path)

        gt_edge_path = os.path.join(cwd, 'dataset', target_dir, 'gt_edge', image_name)
        gt_edge = gt_img.convert('L')
        gt_edge = np.asarray(gt_edge)
        gt_edge = canny(gt_edge, sigma=2, mode='reflect')
        gt_edge = Image.fromarray(gt_edge)
        gt_edge.save(gt_edge_path)

        cropped_edge_path = os.path.join(cwd, 'dataset', target_dir, 'cr_edge', image_name)
        cr_edge = gt_edge.crop((expand_size, 0, cropped_size + expand_size, output_size))
        cr_edge = ImageOps.expand(cr_edge, border=(expand_size, 0, expand_size, 0))
        cr_edge.save(cropped_edge_path)

        bar.update(1)
    bar.close()


if __name__ == "__main__":
    cropped_size = int(sys.argv[1])
    output_size = int(sys.argv[2])
    expand_size = (output_size - cropped_size) // 2
    if len(sys.argv) == 4:
        target_dir = sys.argv[3]
    else:
        target_dir = 'train'

    preprocessing(cropped_size, output_size, expand_size, target_dir)
