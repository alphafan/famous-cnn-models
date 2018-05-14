from pathlib import Path
import os
from datetime import datetime

import numpy as np
import sklearn.utils
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

##########################################################################
# - Load Datasets
# - Train / Test / Validation Splitting
# - Data Shuffling
##########################################################################

# Set logging to Error to ignore some depreciated warnings ...
tf.logging.set_verbosity(tf.logging.ERROR)

print(datetime.now(), 'Downloading MNIST dataset \n')

# Download mnist data to directory
dataDir = os.path.join(str(Path(__file__).absolute().parent.parent), 'mnist')
mnist = input_data.read_data_sets(dataDir, one_hot=True, reshape=False)

# Resume to logging level
tf.logging.set_verbosity(tf.logging.INFO)

# Split data into train / test / validation
X_train, y_train = mnist.train.images, mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels

# Modify mnist input shape into 32 * 32 so that LeNet5 will accept
# Pad two rows of 0s at top and bottom and two columns at left and right
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print()
print(datetime.now(), 'Training data shapes')
print('Input Shape  :', np.shape(X_train))
print('Output Shape :', np.shape(y_train))

# Shuffle train data
print()
print(datetime.now(), 'Shuffling dataset')
X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=0)