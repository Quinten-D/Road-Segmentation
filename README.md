# Road Segmentation
For the second project of the EPFL's Machine Learning master course (CS-433), we tackled the road segmentation challenge.
Given a set of labeled satellite images, we had to construct a classifier to recognise the roads in these sorts of pictures.
(full details: https://www.aicrowd.com/challenges/epfl-ml-road-segmentation)

![intoout](https://user-images.githubusercontent.com/56118785/217513592-665f40af-8a89-4158-a76d-96133d3585fb.png)

We implemented the famous U-Net classifier and came up with some custom networks of our own. After noticing that some of 
the simpler classifiers had a tendency to create "ragged" roads in their segmented images we tried introducing our custom loss function "ragged loss".
Ragged loss penalises classifiers who create irregular roads with ragged edges or holes in their segmented output images.

<p align="center">
  <img width="676" alt="rag" src="https://user-images.githubusercontent.com/56118785/217513974-d48953a0-c958-4ca4-bb34-0b9f0ba0ae34.png">
</p>

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
