"""
This file contains code fragments from the Baseline for machine learning project on road segmentation.
Credits: Aurelien Lucchi, ETH Zürich
"""

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import tensorflow.python.platform
import numpy
#import tensorflow as tf        # switch to older version of tensorflow
#import tensorflow.compat.v1 as tf
import tensorflow as tf
#tf.disable_eager_execution()


PIXEL_DEPTH = 255
TRAINING_SIZE = 32  # Number of training samples, max 1800, Training data = 1 to TRAINING_SIZE
INDEX_OF_FIRST_VALIDATION_SAMPLE = 1751  # Validation data = INDEX_OF_FIRST_VALIDATION_SAMPLE to 1800
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 3
# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16


def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                #im_patch = im[j : j + w, i : i + h]
                im_patch = im[i: i + h, j: j + w]
            else:
                #im_patch = im[j : j + w, i : i + h, :]
                im_patch = im[i:i + h, j: j + w, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, first_index, last_index):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(first_index, last_index + 1):
        #imageid = "satImage_%.3d" % i
        imageid = "satImage_" + str(i)
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)
    return numpy.asarray(imgs)

# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]

def to_one_hot_labels(list):
    return_list = []
    for item in list:
        if item >= 0.7:
            return_list.append([0, 1])
        else:
            return_list.append([1, 0])
    return return_list

# Extract label images
def extract_labels(filename, first_index, last_index):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(first_index, last_index + 1):
        #imageid = "satImage_%.3d" % i
        imageid = "satImage_" + str(i)
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(gt_imgs)
    one_hot_labels = [
        #img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(last_index + 1 - first_index)
        #img_crop(gt_imgs[i], 1, 1) for i in range(last_index + 1 - first_index)
        to_one_hot_labels(numpy.asarray(image).flatten()) for image in gt_imgs
    ]
    labels = numpy.asarray(one_hot_labels)
    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

# Own function to measure accuracy on the validation set
def validation_accuracy(prediction, groundtruth):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(prediction)):
        pred_label = prediction[i]
        groundtruth_label = groundtruth[i]
        if numpy.argmax(pred_label) == 0 and numpy.argmax(groundtruth_label) == 0:
            true_neg += 1
        if numpy.argmax(pred_label) == 0 and numpy.argmax(groundtruth_label) == 1:
            false_neg += 1
        if numpy.argmax(pred_label) == 1 and numpy.argmax(groundtruth_label) == 0:
            false_pos += 1
        if numpy.argmax(pred_label) == 1 and numpy.argmax(groundtruth_label) == 1:
            true_pos += 1
    return (true_pos+true_neg) / len(prediction)

# Own function to help measure F1 on the validation set
def validation_true_pos_false_pos_false_neg(prediction, groundtruth):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(prediction)):
        pred_label = prediction[i]
        groundtruth_label = groundtruth[i]
        if numpy.argmax(pred_label)==0 and numpy.argmax(groundtruth_label)==0:
            true_neg+=1
        if numpy.argmax(pred_label)==0 and numpy.argmax(groundtruth_label)==1:
            false_neg+=1
        if numpy.argmax(pred_label)==1 and numpy.argmax(groundtruth_label)==0:
            false_pos+=1
        if numpy.argmax(pred_label)==1 and numpy.argmax(groundtruth_label)==1:
            true_pos+=1
    return true_pos, false_pos, false_neg

def validate_model(validation_data, validation_labels):
    print("Validating model")
    loaded_model = tf.keras.models.load_model("simple_autoencoder.model", compile=False)
    predictions = loaded_model.predict(validation_data)
    #predictions = predictions.reshape((50, 256, 2))"""
    """loaded_graph = tf.saved_model.load("my_model")
    predictions = loaded_graph(validation_data).numpy()"""
    #print(predictions.shape)
    #validation_labels = validation_labels.reshape((50, 256, 2))
    #print(validation_labels.shape)
    # start computation of f1 and accuracy
    accuracys = []
    true_pos = 0
    false_pos = 0
    false_neg = 0
    #print("image pred")
    #print(predictions[0])
    #print("my loss of first image: " + str(my_loss([validation_labels[0]], [predictions[0]])))
    for i in range(len(predictions)):
        image_prediction = predictions[i]
        ground_truth = validation_labels[i]
        accuracys.append(validation_accuracy(image_prediction, ground_truth))
        true_p, false_p, false_n = validation_true_pos_false_pos_false_neg(image_prediction, ground_truth)
        true_pos += true_p
        false_pos += false_p
        false_neg += false_n
        # print(true_pos, false_pos)
    accuracy_on_unseen_data = numpy.mean(numpy.array(accuracys))
    print("Accuracy on validation data: " + str(accuracy_on_unseen_data))
    # Compute f1
    precision = true_pos / (true_pos + false_pos + 1.)  # +1 to avoid divide by zero
    recall = true_pos / (true_pos + false_neg + 1.)  # +1 to avoid divide by zero
    f1 = 2 * precision * recall / (precision + recall + 0.0000001)  # +0.0000001 to avoid divide by zero
    print("F1 score of validation data: " + str(f1))

    #save some of the predictions
    for counter in range(10):
        gt_img_3c = numpy.zeros((256, 256, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(predictions[counter].reshape((256, 256, 2)))
        for i in range(256):
            for j in range(256):
                no_road = gt_img8[i][j][0]
                road = gt_img8[i][j][1]
                if no_road>=road:
                    #gt_img_3c[i][j] = [gt_img8[i][j][0], gt_img8[i][j][1], 0.]
                    gt_img_3c[i][j] = [1., 0. , 0.]
                if no_road<road:
                    gt_img_3c[i][j] = [0., 1., 0.]
        gt_img_3c = img_float_to_uint8(gt_img_3c)
        label_pic = Image.fromarray(gt_img_3c, 'RGB')
        label_pic.save('validation_prediction_' + str(1751+counter) + '.png')

def my_loss(true_labels, prediction):
    batch_size = tf.shape(prediction)[0]
    print("start my loss computation")
    prediction_probs = tf.nn.softmax(prediction)
    prediction_probs = tf.reshape(prediction_probs, (batch_size, 256, 256, 2))
    one = tf.ones((batch_size, 256, 256, 2))
    right_shifted = tf.roll(prediction_probs, shift=1, axis=2)
    left_shifted = tf.roll(prediction_probs, shift=-1, axis=2)
    up_shifted = tf.roll(prediction_probs, shift=-1, axis=1)
    down_shifted = tf.roll(prediction_probs, shift=1, axis=1)
    nb_of_bad_neighbours = prediction_probs * (
                one - right_shifted + one - left_shifted + one - up_shifted + one - down_shifted)
    #print(tf.reduce_mean(tf.reduce_sum(prediction_probs, [1, 2]), 0))
    #print(tf.reduce_mean(tf.reduce_sum(nb_of_bad_neighbours, [1, 2]), 0))
    bad_pixels = tf.sigmoid(10. * (nb_of_bad_neighbours - 1.75 * one))
    #print(bad_pixels)
    #ragged_loss = tf.reduce_mean(nb_of_bad_neighbours, [0, 1, 2])[0]
    #ragged_loss = tf.reduce_mean(bad_pixels, [0, 1, 2])[0]
    ragged_loss = tf.reduce_mean(tf.reduce_sum(bad_pixels, [1, 2]), 0)[0]
    print("ragged loss")
    print(ragged_loss)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    return loss(true_labels, prediction) + 0.001 * ragged_loss


# Get train data and labels
data_dir = "augmented_training/"
train_data_filename = data_dir + "images/"
train_labels_filename = data_dir + "groundtruth/"
# Extract it into numpy arrays.
train_data = extract_data(train_data_filename, 1, TRAINING_SIZE)
train_labels = extract_labels(train_labels_filename, 1, TRAINING_SIZE)

# Get validation data and labels
validation_data = extract_data(train_data_filename, INDEX_OF_FIRST_VALIDATION_SAMPLE, 1800)
validation_labels = extract_labels(train_labels_filename, INDEX_OF_FIRST_VALIDATION_SAMPLE, 1800)


def run_train_model():
    # Define model
    model = tf.keras.models.Sequential()
    # Encoder
    model.add(tf.keras.layers.Conv2D(input_shape=(256, 256, 3), filters=16, kernel_size=5, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # Decoder
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=5, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=5, padding="same", use_bias=True))  #, activation='relu'
    # Flatten
    model.add(tf.keras.layers.Reshape((65536, 2)))

    # Define loss and optimizer
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
    model.compile(optimizer=opt,
                  loss=my_loss,
                  metrics=['accuracy'])

    # Run the training loop
    model.fit(train_data, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

    # Save the model
    #tf.saved_model.save(model, "my_model")
    model.save("simple_autoencoder.model")


if __name__ == "__main__":
    run_train_model()
    validate_model(validation_data, validation_labels)


