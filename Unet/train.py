import os
import torch

from torch.optim import Adam
from tqdm import tqdm, trange
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from images_dataset import ImagesDataset
from unet import UNet
from helpers import DiceLoss, accuracy_score_tensors, f1_score_tensors, train_test_split

seed = 0

data_path = os.path.abspath("../Data/")
models_path = os.path.abspath("models/")
model_path = os.path.join(models_path, 'unetBest.pt')

data_train_path = os.path.join(data_path, 'augmented_training')
grdTruth_path = os.path.join(data_train_path, 'groundtruth')
image_path = os.path.join(data_train_path, 'images')


class Trainer:
    def __init__(self, model, lossF, optimizer, weights_path, data_loader, lr_scheduler,
                 valid_data_loader=None, threshold=0.25):
        """
        :param model: The NN Model (torch Module)
        :param lossF: The loss function
        :param optimizer: The optimizer function
        :param weights_path: The path to save the model
        :param data_loader:
        :param lr_scheduler:
        :param valid_data_loader: Loader of validation data
        :param threshold:
        """
        self.tqdm = tqdm
        self.trange = trange
        self.lr_scheduler = lr_scheduler

        self.model = model
        self.lossF = lossF
        self.optimizer = optimizer
        self.weights_path = weights_path
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.threshold = threshold

        dataSize = len(self.data_loader.dataset)
        batchSize = self.data_loader.batch_size
        self.train_steps = dataSize // batchSize
        if self.valid_data_loader is not None:
            validDataSize = len(self.valid_data_loader.dataset)
            validBatchSize = self.valid_data_loader.batch_size
            self.valid_steps = validDataSize // validBatchSize

    def predict_labels(self, output):
        """
        :param output: The raw tensor output
        :returns: A binary tensor where we have 1 if the output value was > threshold
        """
        return (output > self.threshold).type(torch.uint8)

    def train_epoch(self, epoch):
        """
        :param epoch: Epoch number
        :returns: [loss, accuracy, f1]
        """
        self.model.train()

        f1 = 0.0
        loss = 0.0
        accuracy = 0.0
        with self.tqdm(self.data_loader, desc=f'Training Epoch {epoch}', leave=False, unit='batch') as tq:
            tq.set_postfix({"loss": loss, "accuracy": accuracy, "f1": f1})
            for data, target in tq:
                self.optimizer.zero_grad()

                output = self.model(data)
                cur_loss = self.lossF(output, target)
                cur_loss.backward()
                self.optimizer.step()

                output = self.predict_labels(output)
                target = self.predict_labels(target)

                accuracy += accuracy_score_tensors(target, output)
                f1 += f1_score_tensors(target, output)
                loss += cur_loss.item()

                tq.set_postfix({"loss": loss, "accuracy": accuracy, "f1": f1})

        return [loss / self.train_steps, accuracy / self.train_steps, f1 / self.train_steps]

    def valid_epoch(self, epoch):
        """
        :param epoch: Epoch number
        :returns: [loss, accuracy, f1]
        """
        self.model.eval()

        f1 = 0.0
        loss = 0.0
        accuracy = 0.0
        with torch.no_grad():
            with self.tqdm(self.valid_data_loader, desc=f'Validation epoch {epoch}',
                           unit='batch', leave=False) as tq:
                tq.set_postfix({"loss": loss, "accuracy": accuracy, "f1": f1})
                for data, target in tq:
                    output = self.model(data)
                    cur_loss = self.lossF(output, target)

                    output = self.predict_labels(output)
                    target = self.predict_labels(target)

                    loss += cur_loss.item()
                    f1 += f1_score_tensors(target, output)
                    accuracy += accuracy_score_tensors(target, output)

                    tq.set_postfix({"loss": loss, "accuracy": accuracy, "f1": f1})

        return [loss / self.valid_steps, accuracy / self.valid_steps, f1 / self.valid_steps]

    def train(self, epochs):
        max_f1 = 0.0
        stats = dict()
        with self.trange(1, epochs + 1, unit='epoch', desc='Training') as t:
            for epoch in t:
                [train_loss, train_accuracy, train_f1] = self.train_epoch(epoch)
                stats.update({"loss": train_loss, "accuracy": train_accuracy, "f1": train_f1})

                if self.valid_data_loader is not None:
                    [valid_loss, valid_accuracy, valid_f1] = self.valid_epoch(epoch)
                    stats.update({"loss": valid_loss, "accuracy": valid_accuracy, "f1": valid_f1})
                    if valid_f1 > max_f1:
                        max_f1 = valid_f1
                        stats['max_f1'] = max_f1
                        torch.save(self.model.state_dict(), self.weights_path)
                else:
                    if train_f1 > max_f1:
                        max_f1 = train_f1
                        stats['max_f1'] = max_f1
                        torch.save(self.model.state_dict(), self.weights_path)

                self.lr_scheduler.step(train_loss)
                t.set_postfix(stats)


def train(batch_size=10, epochs=50, lr=2e-4, split_ratio=0.15, weight_decay=1e-4):
    torch.manual_seed(seed)

    if not os.path.exists(models_path):
        os.mkdir(models_path)

    dataset = ImagesDataset(
        image_dir=image_path,
        grdTruth_dir=grdTruth_path,
        image_transform=transforms.Compose([transforms.ToTensor()]),
        mask_transform=transforms.Compose([transforms.ToTensor()]),
    )

    train_loader = None
    test_loader = None
    if split_ratio == 0:
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
    else:
        train_set, test_set = train_test_split(dataset=dataset, split_ratio=split_ratio)

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        test_loader = DataLoader(
            dataset=test_set,
            shuffle=False,
            num_workers=2,
        )

    model = UNet()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5)

    trainer = Trainer(
        model=model,
        lossF=DiceLoss(),
        optimizer=optimizer,
        weights_path=model_path,
        data_loader=train_loader,
        lr_scheduler=lr_scheduler,
        valid_data_loader=test_loader
    )
    trainer.train(epochs)


if __name__ == '__main__':
    train(split_ratio=0)
