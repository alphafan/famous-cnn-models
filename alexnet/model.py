from datetime import datetime
import tensorflow as tf
from utils.load_image_net import (
    X_train, X_test, X_validation,
    y_train, y_test, y_validation
)


class AlexNet(object):

    def __init__(self):
        # Input & output placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, 227, 227, 3), name='image')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 103), name='label')

    def run(self):
        """
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        # 2nd Layer: Conv (w ReLu) -> Lrn -> Pool with 2 groups
        # 3rd Layer: Conv (w ReLu)
        # 4th Layer: Conv (w ReLu) splitted into two groups
        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        # 7th Layer: FC (w ReLu) -> Dropout
        # 8th Layer: FC and return unscaled activations
        """
        pass
