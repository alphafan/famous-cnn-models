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
        # Training process related params
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def run(self):
        """
        # 1st Layer: Conv (w ReLu) -> Pool -> Normalization
        # 2nd Layer: Conv (w ReLu) -> Pool -> Normalization
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

        # 1st convolutional layer:  Input 227 * 227 *  3    Output 55 * 55 * 96
        #   - a) Convolution        Input 227 * 227 *  3    Output 55 * 55 * 96
        #   - b) Subsampling        Input  55 *  55 * 96    Output 55 * 55 * 96
        #   - c) Normalization      Input  55 *  55 * 96    Output 27 * 27 * 96
        conv_1 = tf.layers.conv2d(self.X, filters=96, kernel_size=11, strides=4, activation=tf.nn.relu)
        pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=3, strides=2)
        norm_1 = tf.nn.local_response_normalization(pool_1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

        # 2nd convolutional layer:  Input 27 * 27 *  96     Output 13 * 13 * 256
        #   - a) Convolution        Input 27 * 27 *  96     Output 27 * 27 * 256
        #   - b) Subsampling        Input 27 * 27 * 256     Output 13 * 13 * 256
        #   - c) Normalization      Input 13 * 13 * 256     Output 13 * 13 * 256
        conv_2 = tf.layers.conv2d(norm_1, filters=256, kernel_size=5, strides=1, activation=tf.nn.relu, padding='SAME')
        pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=3, strides=2)
        norm_2 = tf.nn.local_response_normalization(pool_2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)

        # 3rd convolutional layer:  Input 13 * 13 * 256     Output 13 * 13 * 384
        conv_3 = tf.layers.conv2d(norm_2, filters=384, kernel_size=3, strides=1, activation=tf.nn.relu, padding='SAME')

        # 4th convolutional layer:  Input 13 * 13 * 384     Output 13 * 13 * 384
        conv_4 = tf.layers.conv2d(conv_3, filters=384, kernel_size=3, strides=1, activation=tf.nn.relu, padding='SAME')

        # 5th convolutional layer:  Input 13 * 13 * 384     Output 13 * 13 * 256
        #   - a) Convolution        Input 13 * 13 * 384     Output 13 * 13 * 256
        #   - b) Subsampling        Input 13 * 13 * 256     Output  6 *  6 * 256
        conv_5 = tf.layers.conv2d(conv_4, filters=256, kernel_size=3, strides=1, activation=tf.nn.relu, padding='SAME')
        pool_5 = tf.layers.max_pooling2d(conv_5, pool_size=3, strides=2)

        # stretching out the 5th convolutional layer into a long n-dimensional tensor
        #                           Input  6 *  6 * 256     Output 9216
        flatten = tf.layers.flatten(pool_5)

        # 1st fully connected layer Input 9216              Output 4906
        #   - a) Full Connected     Input 9216              Output 4906
        #   - b) Dropout            Input 4906              Output 4906
        full_1 = tf.layers.dense(flatten, units=4906, activation=tf.nn.tanh)
        drop_1 = tf.nn.dropout(full_1, keep_prob=0.5)

        # 2nd fully connected layer Input 4906              Output 4906
        #   - a) Full Connected     Input 4906              Output 4906
        #   - b) Dropout            Input 4906              Output 4906
        full_2 = tf.layers.dense(drop_1, units=4906, activation=tf.nn.tanh)
        drop_2 = tf.nn.dropout(full_2, keep_prob=0.5)

        # 3rd fully connected layer Input 4906              Output 103
        full_3 = tf.layers.dense(drop_2, units=103, activation=tf.nn.softmax)

        ##########################################################################
        # Backward propagation
        ##########################################################################

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=full_3)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        ##########################################################################
        # Compute accuracy
        ##########################################################################

        corrects = tf.equal(tf.round(full_3), tf.round(self.y))
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        ##########################################################################
        # Train the network
        ##########################################################################

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        print()
        print(datetime.now(), 'Training LeNet5 model')

        for i in range(self.num_epochs):
            for start in range(0, len(X_train), self.batch_size):
                end = start + self.batch_size
                batch_X, batch_y = X_train[start:end], y_train[start:end]
                sess.run(train, feed_dict={
                    self.X: batch_X, self.y: batch_y})
                if end % 50 == 0:
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
                feed_dict={self.X: X_test, self.y: y_test})
            )
        )

        sess.close()


if __name__ == '__main__':
    net = AlexNet()
    net.run()
