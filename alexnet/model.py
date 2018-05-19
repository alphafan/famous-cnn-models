from datetime import datetime
import tensorflow as tf
from utils.load_image_net import (
    X_train, X_test, X_validation,
    y_train, y_test, y_validation
)


class AlexNet(object):

    def __init__(self):
        # Input & output placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1), name='image')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='label')

    def run(self):
        pass
