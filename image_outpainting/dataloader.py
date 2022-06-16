import glob
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OutpaintingDataset(Dataset):
    def __init__(self, folder_path):
        super(OutpaintingDataset).__init__()
        # Get image list
        self.cropped_list = glob.glob(os.path.join(folder_path, 'cropped')+'/*')
        self.gt_list = glob.glob(os.path.join(folder_path, 'gt')+'/*')
        # Calculate len
        self.data_len = len(self.gt_list)
        # Transforms
        self.to_tensor = transforms.ToTensor()
        
    def __getitem__(self, index):
        cropped_image_path = self.cropped_list[index]
        gt_image_path = self.gt_list[index]
        # Open image
        cr_as_im = Image.open(cropped_image_path)
        gt_as_im = Image.open(gt_image_path)
        # Transform image to tensor
        cr_as_tensor = self.to_tensor(cr_as_im)
        gt_as_tensor = self.to_tensor(gt_as_im)
        
        return cr_as_tensor, gt_as_tensor

    def __len__(self):
        return self.data_len  # of how many examples(images?) you have
