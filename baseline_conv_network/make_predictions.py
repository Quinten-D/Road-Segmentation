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
from train_baseline_conv_network import *


if __name__ == "__main__":
    # Restore the saved weights and make predictions for the test data
    main(restore_model=True)
