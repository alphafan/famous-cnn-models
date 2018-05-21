import tensorflow as tf


class GoogLeNet(object):

    def __init__(self):
        # Input & output placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='image')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 103), name='label')

    @staticmethod
    def inception(inputs, conv11_size, conv33_11_size, conv33_size,
                  conv55_11_size, conv55_size, pool11_size):
        """
        An inception cell consists of 4 individual parts and a concat of them

                                Concatenation
                            /     /     \     \
                          /      /       \       \
                        /       /         \         \
                1 * 1 Conv  3 * 3 Conv  5 * 5 Conv  1 * 1 Conv
                    |           |           |          |
                     \          |           |          |
                      \     1 * 1 Conv  1 * 1 Conv  3 * 3 Pool
                        \       |         /       /
                          \     |      /      /
                            \   |    /     /
                              \ |  /   /
                                Inputs

        - Part 1:
            a) 1 * 1 Convolutional
        - Part 2:
            a) 1 * 1 Convolutional
            b) 3 * 3 Convolutional
        - Part 3:
            a) 1 * 1 Convolutional
            b) 5 * 5 Convolutional
        - Part 4:
            a) 3 * 3 Max Pooling
            b) 1 * 1 Convolutional
        """
        conv11 = tf.layers.conv2d(inputs, conv11_size, [1, 1])

        conv33_11 = tf.layers.conv2d(inputs, conv33_11_size, [1, 1])
        conv33 = tf.layers.conv2d(conv33_11, conv33_size, [3, 3])

        conv55_11 = tf.layers.conv2d(inputs, conv55_11_size, [1, 1])
        conv55 = tf.layers.conv2d(conv55_11, conv55_size, [5, 5])

        pool33 = tf.layers.max_pooling2d(inputs, [3, 3], stride=1)
        pool11 = tf.layers.conv2d(pool33, pool11_size, [1, 1])

        return tf.concat([conv11, conv33, conv55, pool11], 3)

    def run(self):
        pass
