import os
import shutil

import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from unet import UNet
from images_dataset import ImagesDataset
from helpers import make_img_overlays, masks_to_submission, submission_to_masks, predict_labels, save_mask_as_image

seed = 0
models_path = os.path.abspath("models/")
model_path = os.path.join(models_path, 'unetBest.pt')

output_path = os.path.abspath("out/")
data_path = os.path.abspath("../Data/")

data_test_images_path = os.path.join(data_path, 'test_set_images')

submission_path = os.path.join(output_path, 'submission')
submission_file_path = os.path.join(output_path, 'submission.csv')
submission_patch_path = os.path.join(output_path, 'submission_patch')
submission_overlay_path = os.path.join(output_path, 'submission_overlay')


class Predictor:
    def __init__(self, model, data_loader):
        """
        :param model: The NN model (torch Module)
        :param data_loader: The data loader of the test set
        """
        shutil.rmtree(submission_path, ignore_errors=True)
        os.makedirs(submission_path, exist_ok=True)

        self.tqdm = tqdm
        self.model = model
        self.data_loader = data_loader
        self.predictions_filenames = list()

    def get_pred_filename(self, index):
        """
        :param index: index of the image
        :returns: prediction filename
        """
        if len(self.data_loader) <= 1000:
            return f'prediction_{index + 1:03d}.png'
        else:
            return f'prediction_{index + 1:04d}.png'

    def predict(self, threshold):
        """Predicts the masks of images.
        :param threshold: Threshold to differentiate between 0 and 1
        """
        self.model.eval()
        with torch.no_grad():
            with self.tqdm(self.data_loader, unit='batch') as tq:
                for i, (data, _) in enumerate(tq):
                    filename = self.get_pred_filename(i)
                    cur_output_path = os.path.join(submission_path, filename)

                    tq.set_description(desc=filename)

                    output = predict_labels(self.model(data), threshold)
                    save_mask_as_image(output, cur_output_path)
                    self.predictions_filenames.append(cur_output_path)


def get_predictions():
    torch.manual_seed(seed)

    for path in (output_path, submission_path, models_path):
        if not os.path.exists(path):
            os.mkdir(path)

    test_set = ImagesDataset(
        image_dir=data_test_images_path,
        image_transform=transforms.Compose([transforms.ToTensor()])
    )
    test_loader = DataLoader(dataset=test_set, num_workers=2)

    model = UNet()
    model.load_state_dict(torch.load(model_path))

    predictor = Predictor(model, test_loader)
    predictor.predict(0.2)

    masks_to_submission(
        submission_path=submission_file_path,
        masks_paths=predictor.predictions_filenames
    )
    submission_to_masks(submission_file_path, 50, submission_patch_path)
    make_img_overlays(data_test_images_path, submission_patch_path, submission_overlay_path)


if __name__ == '__main__':
    get_predictions()
