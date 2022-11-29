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
#from train_conv_network_1 import *
import train_conv_network_1


if __name__ == "__main__":
    train_conv_network_1.RESTORE_MODEL = True  # If True, restore existing model instead of training a new one
    train_conv_network_1.main()