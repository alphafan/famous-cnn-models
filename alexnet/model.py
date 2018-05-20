from datetime import datetime
import tensorflow as tf
from utils.load_image_net import (
    X_train, X_test, X_validation,
    y_train, y_test, y_validation
)


class AlexNet(object):

    def __init__(self):
        # Input & output placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, 227, 227, 3), name='image')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 103), name='label')
        
        # Weight parameters as devised in the original research paper
        self.weights = {
            "wc1": tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.01), name="wc1"),
            "wc2": tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01), name="wc2"),
            "wc3": tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01), name="wc3"),
            "wc4": tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01), name="wc4"),
            "wc5": tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01), name="wc5")
        }
        # Bias parameters as devised in the original research paper
        self.biases = {
            "bc1": tf.Variable(tf.constant(0.0, shape=[96]), name="bc1"),
            "bc2": tf.Variable(tf.constant(1.0, shape=[256]), name="bc2"),
            "bc3": tf.Variable(tf.constant(0.0, shape=[384]), name="bc3"),
            "bc4": tf.Variable(tf.constant(1.0, shape=[384]), name="bc4"),
            "bc5": tf.Variable(tf.constant(1.0, shape=[256]), name="bc5")
        }
        # fully connected layer
        self.fc_layer = lambda x, W, b, name=None: tf.nn.bias_add(tf.matmul(x, W), b)

    def run(self):
        """
        # 1st Layer: Conv (w ReLu) -> Pool -> Normalization
        # 2nd Layer: Conv (w ReLu) -> Lrn -> Pool with 2 groups
        # 3rd Layer: Conv (w ReLu)
        # 4th Layer: Conv (w ReLu) splitted into two groups
        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        # 7th Layer: FC (w ReLu) -> Dropout
        # 8th Layer: FC and return unscaled activations
        """

        ##########################################################################
        # Forward propagation
        ##########################################################################

        # 1st convolutional layer
        conv1 = tf.nn.conv2d(self.X, self.weights["wc1"], strides=[1, 4, 4, 1], padding="SAME", name="conv1")
        conv1 = tf.nn.bias_add(conv1, self.biases["bc1"])
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.local_response_normalization(conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        # 2nd convolutional layer
        conv2 = tf.nn.conv2d(conv1, self.weights["wc2"], strides=[1, 1, 1, 1], padding="SAME", name="conv2")
        conv2 = tf.nn.bias_add(conv2, self.biases["bc2"])
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.local_response_normalization(conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        # 3rd convolutional layer
        conv3 = tf.nn.conv2d(conv2, self.weights["wc3"], strides=[1, 1, 1, 1], padding="SAME", name="conv3")
        conv3 = tf.nn.bias_add(conv3, self.biases["bc3"])
        conv3 = tf.nn.relu(conv3)

        # 4th convolutional layer
        conv4 = tf.nn.conv2d(conv3, self.weights["wc4"], strides=[1, 1, 1, 1], padding="SAME", name="conv4")
        conv4 = tf.nn.bias_add(conv4, self.biases["bc4"])
        conv4 = tf.nn.relu(conv4)

        # 5th convolutional layer
        conv5 = tf.nn.conv2d(conv4, self.weights["wc5"], strides=[1, 1, 1, 1], padding="SAME", name="conv5")
        conv5 = tf.nn.bias_add(conv5, self.biases["bc5"])
        conv5 = tf.nn.relu(conv5)
        conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        # stretching out the 5th convolutional layer into a long n-dimensional tensor
        flatten = tf.layers.flatten(conv5)

        # 1st fully connected layer
        full_1 = tf.layers.dense(flatten, units=4906, activation=tf.nn.tanh)
        full_1 = tf.nn.dropout(full_1, keep_prob=0.5)

        # 2nd fully connected layer
        full_2 = tf.layers.dense(full_1, units=4906, activation=tf.nn.tanh)
        full_2 = tf.nn.dropout(full_2, keep_prob=0.5)

        # 3rd fully connected layer
        full_3 = tf.layers.dense(full_2, units=103, activation=tf.nn.softmax)

        ##########################################################################
        # Backward propagation
        ##########################################################################

        # Return the complete AlexNet model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(tf.shape(full_3), feed_dict={self.X: X_train[:1]}))


if __name__ == '__main__':
    net = AlexNet()
    net.run()
