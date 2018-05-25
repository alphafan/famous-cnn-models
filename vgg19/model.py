import tensorflow as tf
from datetime import datetime
import numpy as np
from utils.load_image_net_224x224x3 import (
    X_train, X_test, X_validation,
    y_train, y_test, y_validation
)


class VGG19(object):

    def __init__(self, learning_rate=0.001, num_epochs=10, batch_size=100):
        self.num_classes = np.shape(y_train)[1]
        # Input & output placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='image')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes), name='label')
        # Training process related params
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def run(self):
        # Block 1
        conv1_1 = tf.layers.conv2d(self.X, 64, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        conv1_2 = tf.layers.conv2d(conv1_1, 64, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1_2, [2, 2], [2, 2], 'same')
        # Block 2
        conv2_1 = tf.layers.conv2d(pool1, 128, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        conv2_2 = tf.layers.conv2d(conv2_1, 128, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2_2, [2, 2], [2, 2], 'same')
        # Block 3
        conv3_1 = tf.layers.conv2d(pool2, 256, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        conv3_2 = tf.layers.conv2d(conv3_1, 256, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        conv3_3 = tf.layers.conv2d(conv3_2, 256, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        conv3_4 = tf.layers.conv2d(conv3_3, 256, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3_4, [2, 2], [2, 2], 'same')
        # Block 4
        conv4_1 = tf.layers.conv2d(pool3, 512, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        conv4_2 = tf.layers.conv2d(conv4_1, 512, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        conv4_3 = tf.layers.conv2d(conv4_2, 512, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        conv4_4 = tf.layers.conv2d(conv4_3, 512, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(conv4_4, [2, 2], [2, 2], 'same')
        # Block 5
        conv5_1 = tf.layers.conv2d(pool4, 512, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        conv5_2 = tf.layers.conv2d(conv5_1, 512, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        conv5_3 = tf.layers.conv2d(conv5_2, 512, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        conv5_4 = tf.layers.conv2d(conv5_3, 512, [3, 3], [2, 2], 'same', activation=tf.nn.relu)
        pool5 = tf.layers.max_pooling2d(conv5_4, [2, 2], [2, 2], 'same')
        # Block 6
        flat6 = tf.layers.flatten(pool5)
        full6_1 = tf.layers.dense(flat6, 4096, activation=tf.nn.relu)
        full6_2 = tf.layers.dense(full6_1, 4096, activation=tf.nn.relu)
        full6_3 = tf.layers.dense(full6_2, self.num_classes)

        ##########################################################################
        # Backward propagation
        ##########################################################################

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=full6_3)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        ##########################################################################
        # Compute accuracy
        ##########################################################################

        corrects = tf.equal(tf.round(full6_3), tf.round(self.y))
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        ##########################################################################
        # Train the network
        ##########################################################################

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        print()
        print(datetime.now(), 'Training GoogLeNet model')

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
        print(datetime.now(), 'Evaluating GoogLeNet model on test set')
        print('\n{} Test Result: Loss: {} Accuracy: {}\n'.format(
            datetime.now(), *sess.run(
                (loss, accuracy),
                feed_dict={self.X: X_test, self.y: y_test}))
        )

        sess.close()


if __name__ == '__main__':
    net = VGG19()
    net.run()
