import os
import re
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm, trange
from PIL import Image
from skimage import io
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, f1_score

output_path = os.path.abspath("out/")


def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg


def reconstruct_from_labels(submission_path, image_id, mask_filename=None):
    h = w = 16
    imgwidth = int(np.ceil(600 / w) * w)
    imgheight = int(np.ceil(600 / h) * h)
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(submission_path, 'r')
    lines = f.readlines()
    image_id_str = f'{image_id:03d}_'

    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j + w, imgwidth)
        ie = min(i + h, imgheight)
        if prediction == 0:
            adata = np.zeros((w, h))
        else:
            adata = np.ones((w, h))

        im[j:je, i:ie] = binary_to_uint8(adata)

    if mask_filename is not None:
        Image.fromarray(im).save(mask_filename)

    return im


def save_mask_as_image(tensor_output, filename):
    """
    :param tensor_output:
    :param filename:
    """
    predictions = torch.squeeze(tensor_output * 255).numpy()
    Image.fromarray(predictions).save(filename)

def predict_labels(output, threshold):
    """
    :param output: The raw tensor output
    :param threshold: The probability threshold
    :returns: A binary tensor where we have 1 if the output value was > threshold
    """
    return (output > threshold).type(torch.uint8)


def submission_to_masks(submission_filename, nb_masks, masks_path=None):
    """
    :returns: Masks that corresponds to the submission
    """
    if masks_path is not None and not os.path.exists(masks_path):
        os.mkdir(masks_path)
    masks = list()
    for i in trange(nb_masks):
        mask_filename = None
        if masks_path is not None:
            mask_name = f'prediction_{i+1:03d}.png'
            mask_filename = os.path.join(masks_path, mask_name)
        mask = reconstruct_from_labels(submission_filename, i+1, mask_filename)
        masks.append(mask)
    return masks


def single_mask_to_submission(mask_filename, foreground_threshold=0.25):
    im = io.imread(os.path.join(output_path, 'submission', mask_filename))

    mask_patch = np.zeros((im.shape[0] // 16, im.shape[1] // 16), np.uint8)
    for j in range(0, im.shape[1], 16):
        for i in range(0, im.shape[0], 16):
            patch = im[i:i + 16, j:j + 16]
            label = int(np.mean(patch) > (foreground_threshold * 255))
            mask_patch[j // 16, i // 16] = label

    img_number = int(re.search(r"\d+", os.path.basename(mask_filename)).group(0))
    for j in range(mask_patch.shape[1]):
        for i in range(mask_patch.shape[0]):
            yield f'{img_number:03d}_{j * 16}_{i * 16},{mask_patch[j, i]}'


def masks_to_submission(submission_path, masks_paths):
    """
    :param submission_path: CSV file path
    :param masks_paths: List of masks paths

    Given masks, creates a csv file
    """
    with open(submission_path, 'w') as f:
        f.write('id,prediction\n')
        for name in tqdm(masks_paths, desc='Creating csv', unit='mask'):
            f.writelines(f'{s}\n' for s in single_mask_to_submission(name, foreground_threshold=0.25))


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, curInput, target):
        curInput = curInput.view(-1)
        target = target.view(-1)
        intersection = (curInput * target).sum()

        num = 2.0 * intersection + 1.0
        denom = curInput.sum() + target.sum() + 1.0

        return 1 - num / denom


def accuracy_score_tensors(target, output):
    """
    :param target: The correct labels (of type torch.Tensor)
    :param output: The predicted labels (of type torch.Tensor)

    :return: Accuracy score in [0,1]
    """
    return accuracy_score(torch.flatten(target), torch.flatten(output), normalize=True)


def f1_score_tensors(target, output):
    """
    :param target: The correct labels (of type torch.Tensor)
    :param output: The predicted labels (of type torch.Tensor)

    :return: F1 score in [0,1]
    """
    return f1_score(torch.flatten(target), torch.flatten(output), zero_division=1)


def train_test_split(dataset, split_ratio):
    """
    :param dataset: A torch dataset
    :param split_ratio: The train/test split ratio

    :returns: A tuple consisting of the train and test datasets
    """
    generator = torch.Generator()
    test_size = int(split_ratio * len(dataset))
    train_size = len(dataset) - test_size
    return random_split(dataset, [train_size, test_size], generator)


def make_img_overlays(image_dir, mask_dir, output_dir):
    """
    :param image_dir: Directory of the images
    :param mask_dir: Directory of the masks
    :param output_dir: Directory for the output

    Create overlays from all the predictions
    """

    def make_img_overlay(image, mask):
        """
        :param image: The image as ndarray
        :param mask: The mas as ndarray
        :returns: The overlay image
        """

        def img_float_to_uint8(img):
            rimg = img - np.min(img)
            rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
            return rimg

        w = image.shape[0]
        h = image.shape[1]
        color_mask = np.zeros((w, h, 3), dtype=np.uint8)
        color_mask[:, :, 0] = mask

        img8 = img_float_to_uint8(image)
        background = Image.fromarray(img8, 'RGB').convert('RGBA')
        overlay = Image.fromarray(color_mask, 'RGB').convert('RGBA')
        new_img = Image.blend(background, overlay, 0.2)
        return new_img

    os.makedirs(output_dir, exist_ok=True)
    image_names = [f'test_{i + 1}/test_{i + 1}.png' for i in range(50)]
    mask_names = sorted(os.listdir(mask_dir))

    for image_name, mask_name in tqdm(
            zip(image_names, mask_names), total=len(image_names)
    ):
        image = np.asarray(Image.open(os.path.join(image_dir, image_name)))
        mask = np.asarray(Image.open(os.path.join(mask_dir, mask_name)))

        oimg = make_img_overlay(image, mask)
        oimg.save(os.path.join(output_dir, mask_name))
