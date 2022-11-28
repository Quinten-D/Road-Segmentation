import numpy as np
from helper_functions import *
from submission_helpers import *
import os
import matplotlib.image as mpimg


def label_to_img(imgwidth, imgheight, labels):
    """
    turn a 1d list of labels for a certain test image into a 2d list
    """
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for row in range(0, imgheight):
        for column in range(0, imgwidth):
            im[row][column] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def save_overlay_images(labels):
    """
    labels is a list containing lists with pixel labels for every test image
    """
    for i in range(0, len(labels)):
        prediction = labels[i]
        segmentation_image = label_to_img(608, 608, prediction)
        # get test images
        #root_dir = os.path.dirname(__file__) + "/../Data/test_set_images/"
        root_dir = os.path.dirname(__file__) + "/Data/test_set_images/"
        imgs = [load_image(root_dir + "test_" + str(i) + "/test_" + str(i) + ".png") for i in range(1, 51)]
        new_img = make_img_overlay(imgs[i], segmentation_image)
        #plt.imshow(new_img)
        #plt.show()
        #mpimg.imsave("segmentation_images/your_file.png", segmentation_image)
        new_img = new_img.convert('RGB')
        new_img.save("segmentation_images/prediction_" + str(i+1) + ".png")