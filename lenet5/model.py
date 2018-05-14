import tensorflow as tf

from utils.load_mnist import (
    X_train, X_test, X_validation,
    y_train, y_test, y_validation
)


class LeNet5(object):

    def __init__(self, num_epochs=10, batch_size=128):
        # Training process related params
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def forwardPropagation(self):
        """ Forward propagation process of LeNet5 model

        Returns:


        # Layer 1:
        #   - a) Convolution:       Input 32 * 32 * 1 , Output 28 * 28 * 6
        #   - b) Subsampling:       Input 28 * 28 * 6 , Output 14 * 14 * 6
        # Layer 2:
        #   - a) Convolution:       Input 14 * 14 * 6 , Output 10 * 10 * 16
        #   - b) Subsampling:       Input 10 * 10 * 16, Output 5  * 5  * 16
        # Layer 3:
        #   - a) Flatten:           Input 5  * 5  * 16, Output 120
        # Layer 4:
        #   - a) Fully Connected:   Input 120         , Output 84
        # Layer 5:
        #   - a) Fully connected:   Input 84          , Output 10
        """

        X = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1), name='image')
        y = tf.placeholder(dtype=tf.float32, shape=(None,), name='label')

        # Layer 1:
        #   - a) Convolution:       Input 32 * 32 * 1 , Output 28 * 28 * 6
        #   - b) Subsampling:       Input 28 * 28 * 6 , Output 14 * 14 * 6
        conv_1 = tf.layers.conv2d(X, filters=2, kernel_size=5)
        with tf.Session() as sess:
            print(sess.run(conv_1))
        # Layer 2:
        #   - a) Convolution:       Input 14 * 14 * 6 , Output 10 * 10 * 16
        #   - b) Subsampling:       Input 10 * 10 * 16, Output 5  * 5  * 16
        # Layer 3:
        #   - a) Flatten:           Input 5  * 5  * 16, Output 120
        # Layer 4:
        #   - a) Fully Connected:   Input 120         , Output 84
        # Layer 5:
        #   - a) Fully connected:   Input 84          , Output 10


net = LeNet5()
net.forwardPropagation()




