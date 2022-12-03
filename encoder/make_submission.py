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
from submission_helpers import *
tf.disable_eager_execution()


if __name__ == "__main__":
    print("Making submission")
    loaded_model = tf.keras.models.load_model("baseline.model")