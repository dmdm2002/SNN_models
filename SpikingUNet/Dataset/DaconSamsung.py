"""https://dacon.io/competitions/official/236132/overview/description"""
import glob

import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
from SpikingUNet.Dataset.OneHot import label_to_one_hot_label


class DaconSamsung(Dataset):
    def __init__(self, root, run_type, transform, infer):
        super().__init__()
        self.run_type = run_type
        self.transform = transform
        self.infer = infer

        if infer:
            self.images = glob.glob(f'{root}/{run_type}/test_image/*.png')
        else:
            self.images = glob.glob(f'{root}/{run_type}/{run_type}_source_image/*.png')
            self.masks = glob.glob(f'{root}/{run_type}/{run_type}_source_gt/*.png')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask = np.array(Image.open(self.masks[idx]))
        mask[mask == 255] = 12

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
