from helper_functions import *
from submission_helpers import *
from overlay_maker import *
import os
import matplotlib.image as mpimg


def predict_labels(image):
    """
    predict for each pixel of the image if it belongs to a road or not
    based on its colour (gray==road)
    :param image: a test image
    :return: a list of labels of lengt heigth*width
    """
    label_list = []
    for row in image:
        for pixel in row:
            (red_value, green_value, blue_value) = pixel
            red_value *= 255.
            green_value *= 255.
            blue_value *= 255.
            if red_value<120. and green_value<120. and blue_value<120. and difference_between_colours_is_small(red_value, green_value, blue_value):
            #if red_value < 100. and green_value < 100. and blue_value < 100. and difference_between_colours_is_small(red_value, green_value, blue_value):
                label_list.append(1.)
            else:
                label_list.append(0.)
    return label_list

def difference_between_colours_is_small(red, green, blue):
    return -10.<(red-green)<10. and -10.<(red-blue)<10. and -10.<(blue-green)<10.
    #return -7. < (red - green) < 7. and -7. < (red - blue) < 7. and -7. < (blue - green) < 7.


if __name__ == "__main__":
    # Load a set of images
    """
    root_dir = os.path.dirname(__file__) + "/../Data/training/"
    image_dir = root_dir + "groundtruth/"
    files = os.listdir(image_dir)
    n = min(20, len(files))  # Load maximum 20 images
    print("Loading " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    print(files[0])
    print(imgs[0][0])
    """

    # Load all test images
    root_dir = os.path.dirname(__file__) + "/../Data/test_set_images/"
    print("Loading test images")
    imgs = [load_image(root_dir + "test_" + str(i) + "/test_" + str(i) + ".png") for i in range(1, 51)]
    #print(imgs[0][0])

    # plot first test image
    """
    cimg = imgs[0]
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(cimg, cmap="Greys_r")
    plt.show()
    """

    # predict labels for each test image
    print("Start making predictions ...")
    labels = []
    for i in range(0, len(imgs)):
        test_image = imgs[i]
        labels.append(predict_labels(test_image))

    # visualise predictions by making overlay png images
    save_overlay_images(labels)

    # make a submission using the predictions
    labels_to_submission("first_submission.csv", labels)