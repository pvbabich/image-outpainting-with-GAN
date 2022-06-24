import glob
import os
import sys

import cv2
import tqdm


def preprocessing(cropped_size, output_size, expand_size, target_dir='train'):

    os.chdir('..')
    cwd = os.getcwd()
    gt_names = glob.glob(os.path.join(cwd, 'dataset', target_dir, 'gt') + '/*.jpg')
    bar = tqdm.tqdm()
    for img_path in gt_names:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (output_size, output_size))
        cv2.imwrite(img_path, img)
        cropped_part = img.copy()[:, expand_size: expand_size + cropped_size, :]
        cropped_img = cv2.copyMakeBorder(
            cropped_part, 
            top=0, 
            bottom=0, 
            left=expand_size, 
            right=expand_size, 
            borderType=cv2.BORDER_CONSTANT,  
            value=0
        )
        cropped_name = os.path.join(cwd, '../dataset', target_dir, 'cropped', img_path.split('/')[-1])
        cv2.imwrite(cropped_name, cropped_img)
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
