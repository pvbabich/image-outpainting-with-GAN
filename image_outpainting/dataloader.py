import glob
import os

from PIL import Image, ImageOps
from skimage.feature import canny
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_grayscale


def mirror_image(image, mode="RGB"):

    left_part = image.crop((0, 0, image.size[0] // 2, image.size[1]))
    right_part = image.crop((image.size[0] // 2, 0, image.size[0], image.size[1]))
    if mode == "RGB":
        result_image = Image.new(mode="RGB", size=image.size)
    else:
        result_image = Image.new(mode="L", size=image.size)
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

        # Mirror
        gt_im = mirror_image(gt_im, "RGB")
        cr_im = mirror_image(cr_im, "RGB")

        # Transform image to tensor
        gt_tensor = self.to_tensor(gt_im)
        cr_tensor = self.to_tensor(cr_im)
        
        return gt_tensor, cr_tensor

    def __len__(self):
        return self.data_len  # of how many examples(images?) you have
