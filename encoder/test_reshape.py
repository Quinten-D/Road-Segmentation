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



first = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
second = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
t = tf.constant([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=2, padding="same", input_shape=(2, 3)))
model.add(tf.keras.layers.Reshape((12,)))
#model.add(tf.keras.layers.Concatenate(axis=2))

print(model.output_shape)

arr = [[[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]],

       [[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]]]

print(model.predict([arr]))