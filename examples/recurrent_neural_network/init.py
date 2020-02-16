"""Initialization script for every notebook.

This script initializes variables and sets up paths and seeds that are common
to every notebook. It should be called in the first notebook cell.
"""

import logging
import os
import sys
import tensorflow as tf
import warnings

# start logging system and set logging level
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("Starting logger")

# The following warnings are from third party libraries and out of our control
warnings.filterwarnings(action='once')
messages = [
    ".*TF Lite has moved.*",
    "the matrix subclass is not the recommended way.*",
    ".*scipy.sparse.sparsetools.*",
    ".*encoding is deprecated.*",
    ".*may indicate binary incompatibility.*",
    ".*np.asscalar.*",
    ".*resolve package from __spec__.*",
    ".*unclosed file.*",
    ".*lbfbgs failed to converge.*",
    ".*simple_save.*"
]

for msg in messages:
    warnings.filterwarnings("ignore", message=msg)

# see https://github.com/tensorflow/tensorflow/issues/27045
if type(tf.contrib) != type(tf):
    tf.contrib._warning = None
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.logging.set_verbosity(tf.logging.ERROR)

RANDOM_SEED = 42
