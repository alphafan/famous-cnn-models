import tensorflow as tf

from utils.load_image_net import (
    X_train, X_test, X_validation,
    y_train, y_test, y_validation
)


class GoogLeNet(object):

    def __init__(self):
        # Input & output placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, 227, 227, 3), name='image')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 103), name='label')

    @staticmethod
    def inception(inputs, conv_11_size, conv_33_reduce_size, conv_33_size,
                  conv_55_reduce_size, conv_55_size, pool_conv_size):
        """ Apply inception processing to input tensor.

        An inception layer consists of 4 individual parts and a final concatenation of them

                       Concatenation                            - Path 1:
                    /     /     \     \                             a) 1 * 1 Convolutional
                  /      /       \       \
                /       /         \         \                   - Path 2:
        1 * 1 Conv  3 * 3 Conv  5 * 5 Conv  1 * 1 Conv              a) 1 * 1 Convolutional
            |           |           |          |                    b) 3 * 3 Convolutional
             \          |           |          |
              \     1 * 1 Conv  1 * 1 Conv  3 * 3 Pool          - Path 3:
                \       |         /       /                         a) 1 * 1 Convolutional
                  \     |      /      /                             b) 5 * 5 Convolutional
                    \   |    /     /
                      \ |  /   /                                - Path 4:
                        Inputs                                      a) 3 * 3 Max Pooling
                                                                    b) 1 * 1 Convolutional

        Args:
            inputs: (Tensor) -- Input tensor
            conv_11_size: (int) -- Output dimension of Path 1 ( conv 1 * 1 )
            conv_33_reduce_size: (int) -- Output dimension of Path 2a ( conv 1 * 1 )
            conv_33_size: (int) -- Output dimension of Path 2b ( conv 3 * 3 )
            conv_55_reduce_size: (int) -- Output dimension of Path 3a ( conv 1 * 1 )
            conv_55_size: (int) -- Output dimension of Path 3b ( conv 5 * 5 )
            pool_conv_size: (int) -- Output dimension of Path 4b ( conv 1 * 1 )

        Returns:
            concat: (Tensor) -- Output tensor of inception layer
        """
        # Path 1
        conv_11 = tf.layers.conv2d(inputs, filters=conv_11_size, kernel_size=[1, 1], padding='same')

        # Path 2
        conv_33_reduce = tf.layers.conv2d(inputs, filters=conv_33_reduce_size, kernel_size=[1, 1], padding='same')
        conv_33 = tf.layers.conv2d(conv_33_reduce, filters=conv_33_size, kernel_size=[3, 3], padding='same')

        # Path 3
        conv_55_reduce = tf.layers.conv2d(inputs, filters=conv_55_reduce_size, kernel_size=[1, 1], padding='same')
        conv_55 = tf.layers.conv2d(conv_55_reduce, filters=conv_55_size, kernel_size=[5, 5], padding='same')

        # Path 4
        pool = tf.layers.max_pooling2d(inputs, pool_size=[3, 3], strides=1, padding='same')
        conv_pool = tf.layers.conv2d(pool, filters=pool_conv_size, kernel_size=[1, 1], padding='same')

        # Concatenation
        return tf.concat([conv_11, conv_33, conv_55, conv_pool], 3)

    def run(self):
        """
        # 1st Convolutional Layer:  Input 227 * 227 *  3 ,    Output  55 *  55 * 64
        #   - a) Convolution        Input 227 * 227 *  3 ,    Output 111 * 111 * 64
        #   - b) Subsampling        Input 111 * 111 * 64 ,    Output  55 *  55 * 64
        #   - c) Normalization      Input  55 *  55 * 64 ,    Output  55 *  55 * 64

        # 2nd Convolutional Layer:  Input  55 * 55 *  64 ,    Output 13 * 13 * 256
        #   - a) Convolution a      Input  55 * 55 *  64 ,    Output 55 * 55 *  64
        #   - b) Convolution b      Input  55 * 55 *  64 ,    Output 55 * 55 * 192
        #   - c) Normalization      Input  55 * 55 * 192 ,    Output 55 * 55 * 192
        #   - d) Subsampling        Input  55 * 55 * 192 ,    Output 26 * 26 * 192

        # 3rd Inception Layer:      Input  26 * 26 * 192 ,    Output 26 * 26 * 480
        #   - a) Inception a        Input  26 * 26 * 192 ,    Output 26 * 26 * 256
        #   - b) Inception b        Input  26 * 26 * 256 ,    Output 26 * 26 * 480
        #   - c) Subsampling        Input  26 * 26 * 480 ,    Output 12 * 12 * 480
        """
        # 1st Convolutional Layer
        conv_1 = tf.layers.conv2d(self.X, filters=64, kernel_size=[7, 7], strides=[2, 2])
        pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=[3, 3], strides=[2, 2])
        norm_1 = tf.nn.local_response_normalization(pool_1, depth_radius=2.0, bias=1.0, alpha=2e-4, beta=0.75)

        # 2nd Convolutional Layer
        conv_2a = tf.layers.conv2d(norm_1, filters=64, kernel_size=[1, 1])
        conv_2b = tf.layers.conv2d(conv_2a, filters=192, kernel_size=[3, 3])
        norm_2 = tf.nn.local_response_normalization(conv_2b, depth_radius=2.0, bias=1.0, alpha=2e-4, beta=0.75)
        pool_2 = tf.layers.max_pooling2d(norm_2, pool_size=[3, 3], strides=[2, 2])

        # 3rd Inception Layer
        inception_3a = self.inception(pool_2, conv_11_size=64,
                                      conv_33_reduce_size=96, conv_33_size=128,
                                      conv_55_reduce_size=16, conv_55_size=32,
                                      pool_conv_size=32)
        inception_3b = self.inception(inception_3a, conv_11_size=128,
                                      conv_33_reduce_size=128, conv_33_size=192,
                                      conv_55_reduce_size=32, conv_55_size=96,
                                      pool_conv_size=64)
        pool_3 = tf.layers.max_pooling2d(inception_3b, pool_size=[3, 3], strides=[2, 2])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            shape = sess.run(tf.shape(pool_3), feed_dict={self.X: X_train[:1]})
            print(shape)


if __name__ == '__main__':
    net = GoogLeNet()
    net.run()
