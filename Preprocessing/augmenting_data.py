import os
import torchvision.transforms.functional as TVT

from PIL import Image
from torchvision.transforms import Resize, TenCrop

initial_size = (400, 400)
augmented_size = (256, 256)

images_path = "../Data/training/images"
groundtruths_path = "../Data/training/groundtruth"

aug_images_path = "../Data/augmented_training/images"
aug_groundtruths_path = "../Data/augmented_training/groundtruth"

def save_augmented_image_mask_pair(cur_image, cur_mask, cur_idx):
    """
    Args:
        cur_image: cur_image of type Image
        cur_mask: cur_mask of type Image
        cur_idx: index of the cur_image/cur_mask pair
    """
    cur_filename = f'satImage_{cur_idx}.png'
    cur_image.save(os.path.join(aug_images_path, cur_filename))
    cur_mask.save(os.path.join(aug_groundtruths_path, cur_filename))


def augment_dataset():
    """
    Generates the augmented dataset which consists of:
    1) For each initial cur_image, 10 images coming from TenCrop (4 corners + center for the cur_image and its mirror)
    2) For each initial cur_image, we rotate by 45 degree 8 times, and center crop for non-multiple of 90 rotations
       Note that this will include the initial cur_image
    """
    for cur_dir in (aug_images_path, aug_groundtruths_path):
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

    images_names = os.listdir(images_path)
    masks_names = os.listdir(groundtruths_path)

    resize = Resize(size=augmented_size)
    ten_crop = TenCrop(size=augmented_size)

    angles = [45 * i for i in range(8)]

    cur_idx = 1
    for i in range(len(images_names)):
        print("Processing image " + str(i))

        cur_image = Image.open(os.path.join(images_path, images_names[i]))
        cur_mask = Image.open(os.path.join(groundtruths_path, masks_names[i]))

        images_ten_cropped = ten_crop(cur_image)
        masks_ten_cropped = ten_crop(cur_mask)
        for j in range(len(images_ten_cropped)):
            save_augmented_image_mask_pair(images_ten_cropped[j], masks_ten_cropped[j], cur_idx)
            cur_idx += 1

        for angle in angles:
            cur_image_rotated = TVT.rotate(cur_image, angle)
            cur_mask_rotated = TVT.rotate(cur_mask, angle)
            if angle % 90 != 0:
                cur_image_rotated = TVT.center_crop(cur_image_rotated, augmented_size)
                cur_mask_rotated = TVT.center_crop(cur_mask_rotated, augmented_size)
            else:
                cur_image_rotated = resize(cur_image_rotated)
                cur_mask_rotated = resize(cur_mask_rotated)
            save_augmented_image_mask_pair(cur_image_rotated, cur_mask_rotated, cur_idx)
            cur_idx += 1

augment_dataset()