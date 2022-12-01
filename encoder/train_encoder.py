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
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 32
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 100
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0
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

    """img_patches = [
        img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(last_index + 1 - first_index)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return numpy.asarray(data)"""
    return numpy.asarray(imgs)

# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]

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
    gt_patches = [
        img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(last_index + 1 - first_index)
        #img_crop(gt_imgs[i], 1, 1) for i in range(last_index + 1 - first_index)
    ]
    """data = numpy.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    labels = numpy.asarray(
        [value_to_class(numpy.mean(data[i])) for i in range(len(data))]
    )"""
    labels = []
    for image in gt_patches:
        #i = image[0]
        #c = numpy.mean(image[0])
        image_labels = [
            value_to_class(numpy.mean(image[i])) for i in range(len(image))
        ]
        labels.append(image_labels)
    labels = numpy.asarray(labels)
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
    """
    prediction_image_patches = #img_crop(prediction_image, IMG_PATCH_SIZE, IMG_PATCH_SIZE)
    groundtruth_image_patches = img_crop(groundtruth_image, IMG_PATCH_SIZE, IMG_PATCH_SIZE)
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(prediction_image_patches)):
        mean_predicted_patch_value = numpy.mean(prediction_image_patches[i])
        mean_groundtruth_patch_value = numpy.mean(groundtruth_image_patches[i])
        if mean_groundtruth_patch_value>foreground_threshold and mean_predicted_patch_value==1:
            true_pos+=1
        elif mean_groundtruth_patch_value<=foreground_threshold and mean_predicted_patch_value==0:
            true_neg+=1
        elif mean_groundtruth_patch_value<=foreground_threshold and mean_predicted_patch_value==1:
            false_pos+=1
        elif mean_groundtruth_patch_value>foreground_threshold and mean_predicted_patch_value==0:
            false_neg+=1
    #precision = true_pos/(true_pos+false_pos)
    #recall = true_pos/(true_pos+false_neg)
    return true_pos, false_pos, false_neg
    """

def validate_model(validation_data, validation_labels):
    print("Validating model")
    loaded_model = tf.keras.models.load_model("baseline.model")
    predictions = loaded_model.predict(validation_data)
    #predictions = predictions.reshape((50, 256, 2))
    print(predictions.shape)
    #validation_labels = validation_labels.reshape((50, 256, 2))
    print(validation_labels.shape)
    # start computation of f1 and accuracy
    accuracys = []
    true_pos = 0
    false_pos = 0
    false_neg = 0
    print("image pred")
    print(predictions[1])
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


# Get test data and labels
data_dir = "augmented_training/"
train_data_filename = data_dir + "images/"
train_labels_filename = data_dir + "groundtruth/"
# Extract it into numpy arrays.
train_data = extract_data(train_data_filename, 1, TRAINING_SIZE)
train_labels = extract_labels(train_labels_filename, 1, TRAINING_SIZE)

# Get validation data and labels
validation_data = extract_data(train_data_filename, 1751, 1800)
validation_labels = extract_labels(train_labels_filename, 1751, 1800)


def run_train_model():
    # Define model
    model = tf.keras.models.Sequential()
    """model.add(tf.keras.layers.Conv2D(input_shape=(16, 16, 3), filters=32, kernel_size=5, activation='relu', padding="same", use_bias=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding="same", use_bias=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu', use_bias=True))
    model.add(tf.keras.layers.Dense(2, use_bias=True))"""
    model.add(tf.keras.layers.Conv2D(input_shape=(256, 256, 3), filters=16, kernel_size=5, activation='relu', padding="same", use_bias=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding="same", use_bias=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding="same", use_bias=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding="same", use_bias=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=5, activation='relu', padding="same", use_bias=True))
    model.add(tf.keras.layers.Reshape((256, 2)))

    """def my_loss(true_labels, prediction):
        print(prediction.shape)
        prediction_list_form = tf.concat([prediction[0], prediction[1]], 2)
        print("kjkjk")
        print(prediction_list_form.shape)
        prediction_list_form = tf.reshape(prediction_list_form, (256, 2))
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        return loss(true_labels, prediction_list_form)"""

    # Define loss and optimizer
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
    model.compile(optimizer=opt,
                  loss=loss_function,
                  metrics=['accuracy'])

    # Run the training loop
    model.fit(train_data, train_labels, epochs=3, batch_size=16, shuffle=True)

    # Save the model
    model.save("baseline.model")


if __name__ == "__main__":
    #print(train_data.shape)
    #print(train_labels.shape)
    run_train_model()

    validate_model(validation_data , validation_labels)



    """
    print(train_data.shape)
    print(train_labels.shape)
    pic = img_float_to_uint8(train_data[13])
    pic = Image.fromarray(pic, 'RGB')
    pic.save('test0.png')

    gt_img_3c = numpy.zeros((16, 16, 3), dtype=numpy.uint8)
    gt_img8 = img_float_to_uint8(train_labels[13].reshape((16, 16, 2)))
    #gt_img_3c[:, :, 0] = gt_img8
    #gt_img_3c[:, :, 1] = gt_img8
    #gt_img_3c[:, :, 2] = gt_img8
    for i in range(16):
        for j in range(16):
            gt_img_3c[i][j] = [gt_img8[i][j][0], gt_img8[i][j][1], 0.]
    gt_img_3c = img_float_to_uint8(gt_img_3c)
    label_pic = Image.fromarray(gt_img_3c, 'RGB')
    label_pic.save('label0.png')

    #label_pic = img_float_to_uint8(train_labels[0].reshape((16, 16, 2)))
    #label_pic = Image.fromarray(label_pic, 'RGB')
    #label_pic.save('label0.png')
    """
