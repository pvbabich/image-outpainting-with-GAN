import glob
import os

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


def mirror_image(image):

    left_part = image.crop((0, 0, image.size[0] // 2, image.size[1]))
    right_part = image.crop((image.size[0] // 2, 0, image.size[0], image.size[1]))
    result_image = Image.new(mode="RGB", size=image.size)
    result_image.paste(right_part, (0, 0))
    result_image.paste(left_part, (image.size[0] // 2, 0))
    return result_image

class OutpaintingDataset(Dataset):
    def __init__(self, folder_path):
        super(OutpaintingDataset).__init__()
        # Get image list
        self.gt_list = glob.glob(os.path.join(folder_path, 'gt')+'/*.jpg')
        self.cr_list = glob.glob(os.path.join(folder_path, 'cropped') + '/*.jpg')
        # Calculate len
        self.data_len = len(self.gt_list)
        # Transforms
        self.to_tensor = transforms.ToTensor()
        
    def __getitem__(self, index):

        gt_image_path = self.gt_list[index]
        cr_image_path = self.cr_list[index]

        # Open image
        gt_im = Image.open(gt_image_path)
        cr_im = Image.open(cr_image_path)
        expand_size = (gt_im.size[0] - cr_im.size[0]) // 2
        cr_im = ImageOps.expand(cr_im, border=(expand_size, 0, expand_size, 0))

        # Mirror
        gt_im = mirror_image(gt_im)
        cr_im = mirror_image(cr_im)

        # Transform image to tensor
        gt_tensor = self.to_tensor(gt_im)
        cr_tensor = self.to_tensor(cr_im)
        
        return gt_tensor, cr_tensor

    def __len__(self):
        return self.data_len  # of how many examples(images?) you have
