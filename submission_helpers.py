#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.image as mpimg
import re
from PIL import Image
import math
import sys


#foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
foreground_threshold = 0.5

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def labels_to_submission_strings(labels, number):
    """Gets a list (1d) of pixel labels for a test image and outputs the strings that should go into the submission file"""
    patch_size = 16
    labels_2d = np.array(labels).reshape((608, 608))
    for j in range(0, 608, patch_size):
        for i in range(0, 608, patch_size):
            patch = labels_2d[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(number, j, i, label))

def labels_to_submission(submission_filename, all_test_labels):
    """Converts big 1d list of predicted labels into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        #for fn in image_filenames[0:]:
        for count, labels in enumerate(all_test_labels):
            f.writelines('{}\n'.format(s) for s in labels_to_submission_strings(labels, count+1))

# functions to help visualize submission
# Convert an array of binary labels to a uint8
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def reconstruct_from_labels(image_id, label_file):
    #label_file = 'dummy_submission.csv'
    h = 16
    w = h
    imgwidth = int(math.ceil((600.0 / w)) * w)
    imgheight = int(math.ceil((600.0 / h)) * h)
    nc = 3

    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(label_file)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
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

        je = min(j+w, imgwidth)
        ie = min(i+h, imgheight)
        if prediction == 0:
            adata = np.zeros((w,h))
        else:
            adata = np.ones((w,h))

        im[j:je, i:ie] = binary_to_uint8(adata)

    Image.fromarray(im).save('submission_predictions/prediction_' + '%.3d' % image_id + '.png')

    return im

