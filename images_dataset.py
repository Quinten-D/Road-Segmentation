import os
from PIL import Image
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, image_dir, grdTruth_dir=None, image_transform=None, mask_transform=None):
        """
        :param image_dir: Path of images
        :param grdTruth_dir: Path of groundtruth images  (None for test datasets)
        :param image_transform: Transform to be applied to images
        :param mask_transform: Transform to be applied to masks
        """
        self.image_dir = image_dir
        self.grdTruth_dir = grdTruth_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        if self.grdTruth_dir is None:
            self.images_names = [f'test_{i + 1}/test_{i + 1}.png' for i in range(50)]
            self.masks_names = list()
        else:
            self.images_names = sorted(os.listdir(self.image_dir))
            self.masks_names = sorted(os.listdir(self.grdTruth_dir))

    def __getitem__(self, idx):
        """
        :param idx: Index of the image
        :returns: (image[idx], mask[idx]) where mask[idx] = 0 if it doesn't exist
        """
        image = Image.open(os.path.join(self.image_dir, self.images_names[idx]))
        if self.image_transform is not None:
            image = self.image_transform(image)

        mask = 0
        if self.grdTruth_dir:
            mask = Image.open(os.path.join(self.grdTruth_dir, self.masks_names[idx]))
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.images_names)
