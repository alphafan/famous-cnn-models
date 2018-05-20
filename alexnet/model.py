from datetime import datetime
import tensorflow as tf
from utils.load_image_net import (
    X_train, X_test, X_validation,
    y_train, y_test, y_validation
)


class AlexNet(object):

    def __init__(self, learning_rate=0.001, num_epochs=10, batch_size=100):
        # Input & output placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, 227, 227, 3), name='image')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 103), name='label')

        # Weight parameters as devised in the original research paper
        self.weights = {
            "wc1": tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.01), name="wc1"),
            "wc2": tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01), name="wc2"),
            "wc3": tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01), name="wc3"),
            "wc4": tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01), name="wc4"),
            "wc5": tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01), name="wc5"),
            "wf1": tf.Variable(tf.truncated_normal([9216, 4096], stddev=0.01), name="wf1"),
            "wf2": tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.01), name="wf2"),
            "wf3": tf.Variable(tf.truncated_normal([4096, 103], stddev=0.01), name="wf3")
        }

        # Bias parameters as devised in the original research paper
        self.biases = {
            "bc1": tf.Variable(tf.constant(0.0, shape=[96]), name="bc1"),
            "bc2": tf.Variable(tf.constant(1.0, shape=[256]), name="bc2"),
            "bc3": tf.Variable(tf.constant(0.0, shape=[384]), name="bc3"),
            "bc4": tf.Variable(tf.constant(1.0, shape=[384]), name="bc4"),
            "bc5": tf.Variable(tf.constant(1.0, shape=[256]), name="bc5"),
            "bf1": tf.Variable(tf.constant(1.0, shape=[4096]), name="bf1"),
            "bf2": tf.Variable(tf.constant(1.0, shape=[4096]), name="bf2"),
            "bf3": tf.Variable(tf.constant(1.0, shape=[103]), name="bf3")
        }

        # fully connected layer
        self.fc_layer = lambda x, W, b, name=None: tf.nn.bias_add(tf.matmul(x, W), b)

        # Training process related params
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def run(self):
        """
        Run forward/backward propagation on AlexNet
        Get 96.7% accuracy.

        # 1st convolutional layer:  Input 227 * 227 *  3,    Output 55 * 55 * 96
        #   - a) Convolution        Input 227 * 227 *  3,    Output 55 * 55 * 96
        #   - b) Normalization      Input  55 *  55 * 96,    Output 55 * 55 * 96
        #   - c) Subsampling        Input  55 *  55 * 96,    Output 27 * 27 * 96

        # 2nd convolutional layer:  Input 27 * 27 *  96 ,     Output 13 * 13 * 256
        #   - a) Convolution        Input 27 * 27 *  96 ,     Output 27 * 27 * 256
        #   - b) Subsampling        Input 27 * 27 * 256 ,     Output 13 * 13 * 256
        #   - c) Normalization      Input 13 * 13 * 256 ,     Output 13 * 13 * 256

        # 3rd convolutional layer:  Input 13 * 13 * 256 ,     Output 13 * 13 * 384

        # 4th convolutional layer:  Input 13 * 13 * 384 ,     Output 13 * 13 * 384

        # 5th convolutional layer:  Input 13 * 13 * 384 ,     Output 13 * 13 * 256
        #   - a) Convolution        Input 13 * 13 * 384 ,     Output 13 * 13 * 256
        #   - b) Subsampling        Input 13 * 13 * 256 ,     Output  6 *  6 * 256

        # Flattern layer:           Input  6 *  6 * 256 ,     Output 9216

        # 1st fully connected layer Input 9216          ,    Output 4906
        #   - a) Full Connected     Input 9216          ,    Output 4906
        #   - b) Dropout            Input 4906          ,    Output 4906

        # 2nd fully connected layer Input 4906          .    Output 4906
        #   - a) Full Connected     Input 4906          ,    Output 4906
        #   - b) Dropout            Input 4906          ,    Output 4906

        # 3rd fully connected layer Input 4906          ,    Output 103
        """

        ##########################################################################
        # Forward propagation
        ##########################################################################

        # img = tf.reshape(self.X, [-1, 227, 227, 3])

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
        fc1 = self.fc_layer(flatten, self.weights["wf1"], self.biases["bf1"], name="fc1")
        fc1 = tf.nn.tanh(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

        # 2nd fully connected layer
        fc2 = self.fc_layer(fc1, self.weights["wf2"], self.biases["bf2"], name="fc2")
        fc2 = tf.nn.tanh(fc2)
        fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

        # 3rd fully connected layer
        fc3 = self.fc_layer(fc2, self.weights["wf3"], self.biases["bf3"], name="fc3")
        fc3 = tf.nn.softmax(fc3)

        ##########################################################################
        # Backward propagation
        ##########################################################################

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=fc3)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        ##########################################################################
        # Compute accuracy
        ##########################################################################

        corrects = tf.equal(tf.round(fc3), tf.round(self.y))
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        ##########################################################################
        # Train the network
        ##########################################################################

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        print()
        print(datetime.now(), 'Training AlexNet model')

        for i in range(self.num_epochs):
            for start in range(0, len(X_train), self.batch_size):
                end = start + self.batch_size
                batch_X, batch_y = X_train[start:end], y_train[start:end]
                sess.run(train, feed_dict={
                    self.X: batch_X, self.y: batch_y})
                if end % 100 == 0:
                    print('{} Training Epoch {} {}/{}'.format(datetime.now(), i, end, len(X_train)))

            # Show loss on validation dataset when finishing each epoch
            print('\n{} Training Epoch: {} Loss: {} Accuracy: {}\n'.format(
                datetime.now(), i,
                *sess.run(
                    (loss, accuracy),
                    feed_dict={self.X: X_validation, self.y: y_validation}))
            )

        ##########################################################################
        # Compute accuracy on test set
        ##########################################################################

        # Evaluate on test set
        print(datetime.now(), 'Evaluating AlexNet model on test set')
        print('\n{} Test Result: Loss: {} Accuracy: {}\n'.format(
            datetime.now(), *sess.run(
                (loss, accuracy),
                feed_dict={self.X: X_test, self.y: y_test}))
        )

        sess.close()


if __name__ == '__main__':
    net = AlexNet()
    net.run()
