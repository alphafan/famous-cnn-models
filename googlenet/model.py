import tensorflow as tf


class GoogLeNet(object):

    def __init__(self):
        # Input & output placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='image')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 103), name='label')

    @staticmethod
    def inception(inputs, conv_11_size, conv_33_reduce_size, conv_33_size,
                  conv_55_reduce_size, conv_55_size, pool_conv_size):
        """ Apply inception processing to input tensor.

        An inception layer consists of 4 individual parts and a Concatenation of them

                       Concatenation                            - Part 1:
                    /     /     \     \                             a) 1 * 1 Convolutional
                  /      /       \       \
                /       /         \         \                   - Part 2:
        1 * 1 Conv  3 * 3 Conv  5 * 5 Conv  1 * 1 Conv              a) 1 * 1 Convolutional
            |           |           |          |                    b) 3 * 3 Convolutional
             \          |           |          |
              \     1 * 1 Conv  1 * 1 Conv  3 * 3 Pool          - Part 3:
                \       |         /       /                         a) 1 * 1 Convolutional
                  \     |      /      /                             b) 5 * 5 Convolutional
                    \   |    /     /
                      \ |  /   /                                - Part 4:
                        Inputs                                      a) 3 * 3 Max Pooling
                                                                    b) 1 * 1 Convolutional

        Args:
            inputs: (Tensor) -- Input tensor
            conv_11_size: (int) -- Output dimension of part 1 ( conv 1 * 1 )
            conv_33_reduce_size: (int) -- Output dimension of patt 2a ( conv 1 * 1 )
            conv_33_size: (int) -- Output dimension of patt 2b ( conv 3 * 3 )
            conv_55_reduce_size: (int) -- Output dimension of patt 3a ( conv 1 * 1 )
            conv_55_size: (int) -- Output dimension of patt 3b ( conv 5 * 5 )
            pool_conv_size: (int) -- Output dimension of patt 4b ( conv 1 * 1 )

        Returns:
            concat: (Tensor) -- Output tensor of inception layer
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
        conv_pool = tf.layers.conv2d(pool, pool_conv_size, [1, 1])

        # Concatenation
        return tf.concat([conv_11, conv_33, conv_55, conv_pool], 3)

    def run(self):
        # 1st Convolutional Layer
        conv_1 = tf.layers.conv2d(self.X, filters=64, keenel_size=[7, 7], stride=2)
        pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=[3, 3])
        norm_1 = tf.nn.local_response_normalization(pool_1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
        pass
