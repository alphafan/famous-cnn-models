import tensorflow as tf


class GoogLeNet(object):

    def __init__(self):
        # Input & output placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='image')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 103), name='label')

    @staticmethod
    def inception(inputs, conv_11_size, conv_33_reduce_size, conv_33_size,
                  conv_55_reduce_size, conv_55_size, pool_size):
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
        # Part 1
        conv_11 = tf.layers.conv2d(inputs, conv_11_size, [1, 1])

        # Part 2
        conv_33_reduce = tf.layers.conv2d(inputs, conv_33_reduce_size, [1, 1])
        conv_33 = tf.layers.conv2d(conv_33_reduce, conv_33_size, [3, 3])

        # Part 3
        conv_55_reduce = tf.layers.conv2d(inputs, conv_55_reduce_size, [1, 1])
        conv_55 = tf.layers.conv2d(conv_55_reduce, conv_55_size, [5, 5])

        # Part 4
        pool = tf.layers.max_pooling2d(inputs, [3, 3], stride=1)
        conv_pool = tf.layers.conv2d(pool, pool_size, [1, 1])

        # Concatenation
        return tf.concat([conv_11, conv_33, conv_55, conv_pool], 3)

    def run(self):
        pass
