from datetime import datetime
import tensorflow as tf

from utils.load_mnist import (
    X_train, X_test, X_validation,
    y_train, y_test, y_validation
)


class LeNet5(object):

    def __init__(self, learning_rate=0.001, num_epochs=10, batch_size=100):
        # Input & output placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1), name='image')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='label')
        # Training process related params
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def run(self):
        """ Forward propagation process of LeNet5 model

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

        ##########################################################################
        # Forward propagation
        ##########################################################################

        print()
        print(datetime.now(), 'Building LeNet5 model')

        # ==> Layer 1:
        #   - a) Convolution:       Input 32 * 32 * 1 , Output 28 * 28 * 6
        #   - b) Subsampling:       Input 28 * 28 * 6 , Output 14 * 14 * 6
        conv_1 = tf.layers.conv2d(self.X, filters=6, kernel_size=5)
        pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=2, strides=2)

        # ==> Layer 2:
        #   - a) Convolution:       Input 14 * 14 * 6 , Output 10 * 10 * 16
        #   - b) Subsampling:       Input 10 * 10 * 16, Output 5  * 5  * 16
        conv_2 = tf.layers.conv2d(pool_1, filters=16, kernel_size=5)
        pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=2, strides=2)

        # Flatten
        #   - a) Flatten:           Input 5  * 5  * 16, Output 400
        flatten = tf.layers.flatten(pool_2)

        # ==> Layer 3:
        #   - a) Fully Connected:   Input 400         , Output 120
        full_1 = tf.layers.dense(flatten, units=120, activation=tf.nn.tanh)

        # ==> Layer 4:
        #   - a) Fully Connected:   Input 120         , Output 84
        full_2 = tf.layers.dense(full_1, units=84, activation=tf.nn.tanh)

        # ==> Layer 5:
        #   - a) Fully connected:   Input 84          , Output 10
        full_3 = tf.layers.dense(full_2, units=10, activation=tf.nn.sigmoid)

        ##########################################################################
        # Backwards propagation
        ##########################################################################

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=full_3)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        ##########################################################################
        # Compute accuracy
        ##########################################################################

        corrects = tf.equal(tf.argmax(full_3, 1), tf.argmax(self.y, 1))
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
                if end % 5000 == 0:
                    print('{} Training Epoch {} {}/{}'.format(datetime.now(), i, end, len(X_train)))

            # Show loss on validation dataset when finishing each epoch
            print('\n{} Training Epoch: {} Loss: {} Accuracy: {}\n'.format(
                datetime.now(), i,
                *sess.run(
                    (loss, accuracy),
                    feed_dict={self.X: X_validation, self.y: y_validation})
            )
            )

        ##########################################################################
        # Compute accuracy on test set
        ##########################################################################

        # Evaluate on test set
        print(datetime.now(), 'Evaluating LeNet5 model on test set')
        print('\n{} Test Result: Loss: {} Accuracy: {}\n'.format(
            datetime.now(), *sess.run(
                (loss, accuracy),
                feed_dict={self.X: X_test, self.y: y_test})
        )
        )

        sess.close()


if __name__ == '__main__':
    net = LeNet5()
    net.run()
