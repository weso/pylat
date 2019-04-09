import os
import tensorflow as tf

from abc import ABC, abstractmethod
from tensorflow.python.saved_model import tag_constants
from src.neuralnet.utils import lazy_property


class BaseNeuralNetwork(ABC):

    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        self.save_path = os.path.join(self.save_dir, 'best.ckpt')
        self.session = None
        self._x_t = None
        self._y_t = None
        self._logits = None

    def init_model(self, X, y):
        self.session = tf.Session()
        self.prediction()
        self.optimize()
        self.error()

    def save(self, save_path):
        inputs = {"x_t": self._x_t}
        outputs = {"pred_proba": self._y_proba}
        tf.saved_model.simple_save(self.session, save_path, inputs, outputs)

    def restore(self, save_path):
        graph = tf.Graph()
        self.session = tf.Session(graph=graph)
        tf.saved_model.loader.load(
            self.session,
            [tag_constants.SERVING],
            save_path,
        )
        self._x_t = graph.get_tensor_by_name('x_input:0')
        self._y_proba = graph.get_tensor_by_name('dnn/y_proba:0')

    @lazy_property
    @abstractmethod
    def prediction(self):
        pass

    @lazy_property
    @abstractmethod
    def optimize(self):
        pass

    @lazy_property
    @abstractmethod
    def error(self):
        pass
