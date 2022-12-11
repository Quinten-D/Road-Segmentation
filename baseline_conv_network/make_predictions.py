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
import submission_helpers
import code
import tensorflow.python.platform
import numpy
#import tensorflow as tf        # switch to older version of tensorflow
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from train_baseline_conv_network import *


RESTORE_MODEL = True  # If True, restore existing model instead of training a new one


def main():  # pylint: disable=unused-argument

    data_dir = "training/"
    train_data_filename = data_dir + "images/"
    train_labels_filename = data_dir + "groundtruth/"

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

    num_epochs = NUM_EPOCHS

    c0 = 0  # bgrd
    c1 = 0  # road
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))

    print("Balancing training data...")
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(train_data.shape)
    train_data = train_data[new_indices, :, :, :]
    train_labels = train_labels[new_indices]

    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32, shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS)
    )
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    train_all_data_node = tf.constant(train_data)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal(
            [5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED  # 5x5 filter, depth 32.
        )
    )
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED)
    )
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal(
            [int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
            stddev=0.1,
            seed=SEED,
        )
    )
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED)
    )
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx=0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value * PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image
    def get_prediction(img):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        img_prediction = label_to_img(
            img.shape[0],
            img.shape[1],
            IMG_PATCH_SIZE,
            IMG_PATCH_SIZE,
            output_prediction,
        )

        return img_prediction

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(filename, image_idx):

        if image_idx == -1:
            image_filename = filename
        else:
            imageid = "satImage_%.3d" % image_idx
            image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)

        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx):

        if image_idx == -1:
            image_filename = filename
        else:
            imageid = "satImage_%.3d" % image_idx
            image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(
            relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )

        conv2 = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )

        # Uncomment these lines to check the size of each layer
        #print('data ' + str(data.get_shape()))
        #print('conv ' + str(conv.get_shape()))
        #print('relu ' + str(relu.get_shape()))
        #print('pool ' + str(pool.get_shape()))
        #print('pool2 ' + str(pool2.get_shape()))

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]
        )
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        # if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases

        if train:
            summary_id = "_0"
            s_data = get_image_summary(data)
            tf.summary.image("summary_data" + summary_id, s_data, max_outputs=3)
            s_conv = get_image_summary(conv)
            tf.summary.image("summary_conv" + summary_id, s_conv, max_outputs=3)
            s_pool = get_image_summary(pool)
            tf.summary.image("summary_pool" + summary_id, s_pool, max_outputs=3)
            s_conv2 = get_image_summary(conv2)
            tf.summary.image("summary_conv2" + summary_id, s_conv2, max_outputs=3)
            s_pool2 = get_image_summary(pool2)
            tf.summary.image("summary_pool2" + summary_id, s_pool2, max_outputs=3)
        return out

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)  # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=train_labels_node, logits=logits
        )
    )

    tf.summary.scalar("loss", loss)

    all_params_node = [
        conv1_weights,
        conv1_biases,
        conv2_weights,
        conv2_biases,
        fc1_weights,
        fc1_biases,
        fc2_weights,
        fc2_biases,
    ]
    all_params_names = [
        "conv1_weights",
        "conv1_biases",
        "conv2_weights",
        "conv2_biases",
        "fc1_weights",
        "fc1_biases",
        "fc2_weights",
        "fc2_biases",
    ]
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)

    # L2 regularization for the fully connected parameters.
    regularizers = (
        tf.nn.l2_loss(fc1_weights)
        + tf.nn.l2_loss(fc1_biases)
        + tf.nn.l2_loss(fc2_weights)
        + tf.nn.l2_loss(fc2_biases)
    )
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,  # Decay step.
        0.95,  # Decay rate.
        staircase=True,
    )
    # tf.scalar_summary('learning_rate', learning_rate)
    tf.summary.scalar("learning_rate", learning_rate)

    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.0).minimize(
        loss, global_step=batch
    )

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:

        if RESTORE_MODEL:
            # Restore variables from disk.
            #saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            saver.restore(s, "stored_weights/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=s.graph)

            print("Initialized!")
            # Loop through training steps.
            print(
                "Total number of iterations = "
                + str(int(num_epochs * train_size / BATCH_SIZE))
            )

            training_indices = range(train_size)

            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)

                steps_per_epoch = int(train_size / BATCH_SIZE)

                for step in range(steps_per_epoch):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset : (offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {
                        train_data_node: batch_data,
                        train_labels_node: batch_labels,
                    }

                    if step == 0:
                        summary_str, _, l, lr, predictions = s.run(
                            [
                                summary_op,
                                optimizer,
                                loss,
                                learning_rate,
                                train_prediction,
                            ],
                            feed_dict=feed_dict,
                        )
                        summary_writer.add_summary(
                            summary_str, iepoch * steps_per_epoch
                        )
                        summary_writer.flush()

                        print("Epoch %d" % iepoch)
                        print("Minibatch loss: %.3f, learning rate: %.6f" % (l, lr))
                        print(
                            "Minibatch error: %.1f%%"
                            % error_rate(predictions, batch_labels)
                        )

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict,
                        )

                # Save the variables to disk.
                #save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                save_path = saver.save(s, "stored_weights/model.ckpt")
                print("Model saved in file: %s" % save_path)

        # If you trained the model
        if not RESTORE_MODEL:
            print("Running prediction on training set")
            prediction_training_dir = "predictions_training/"
            if not os.path.isdir(prediction_training_dir):
                os.mkdir(prediction_training_dir)
            for i in range(1, TRAINING_SIZE + 1):
            #for i in range(1, 2):
                pimg = get_prediction_with_groundtruth(train_data_filename, i)
                Image.fromarray(pimg).save(
                    prediction_training_dir + "prediction_" + str(i) + ".png"
                )
                oimg = get_prediction_with_overlay(train_data_filename, i)
                oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")

        if RESTORE_MODEL:
            print("Running prediction on testing set")
            prediction_testing_dir = "predictions_testing/"
            if not os.path.isdir(prediction_testing_dir):
                os.mkdir(prediction_testing_dir)
            for i in range(1, 5 + 1):
                # Save combination
                pimg = get_prediction_with_groundtruth(
                    "test_set_images/test_" + str(i) + "/test_" + str(i) + ".png", -1)
                Image.fromarray(pimg).save(prediction_testing_dir + "prediction_" + str(i) + ".png")
                # Save overlay
                oimg = get_prediction_with_overlay("test_set_images/test_" + str(i) + "/test_" + str(i) + ".png",
                                                   -1)
                oimg.save(prediction_testing_dir + "overlay_" + str(i) + ".png")
            # List to make submission with
            all_labels = []
            for i in range(1, 50 + 1):
                image_name = "test_set_images/test_" + str(i) + "/test_" + str(i) + ".png"
                one_dimensional_label_list = get_prediction(mpimg.imread(image_name)).ravel()
                all_labels.append(one_dimensional_label_list)
            # Now make final csv submission file
            submission_helpers.labels_to_submission("baseline_submission.csv", all_labels)


if __name__ == "__main__":
    #tf.app.run()
    main()