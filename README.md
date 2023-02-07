# Machine Learning - Project 2
This repository contains code for the Road Segmentation challenge, a second project of EPFL's machine learning course.

## Reproduce our results
To reproduce our results, first make sure you have all the dependencies installed (as seen in `requirements.txt`).

Then, just execute `run.py`. This will use a pre-trained model found in the models directory. The submission output will be in the `out` directory.

## Unet
The best results were achieved using UNet. Its implementation is contained in the following files:

- `helpers.py`: Helper functions
- `images_dataset.py`: A class to make manipulation the dataset more convenient
- `predict.py`: Code used to get predictions given a model
- `train.py`: Code used to train a new model
- `unet.py`: U-Net implementation

## Baseline
The directory `baseline_conv_network` holds the provided baseline for the challenge

## Preprocessing
The directory `Preprocessing` holds `augmenting_data.py` which was used to generate the augmented images

## Custom Networks
We've implemented two additional custom networks:

- `encoder`: a network that only utilises the encoder instead of the traditional encoder-decoder architecture
- `simple_encoder-decoder`: a simple encoder-decoder network, equipped with custom loss function ragged loss
