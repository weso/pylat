import os
import tensorflow as tf

from abc import ABC, abstractmethod


class BaseNeuralNetwork(ABC):
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        self.save_path = os.path.join(self.save_dir, 'best.ckpt')
        self.session = None
        self.x_t = None
        self.y_t = None
        self._logits = None
        self.init = None

    def init_model(self, X, y):
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.prediction
        self.optimize
        self.error
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)
        self.additional_inits()

    @abstractmethod
    def save(self, save_path):
        pass

    @abstractmethod
    def restore(self, save_path):
        pass

    def additional_inits(self):
        pass

    @property
    @abstractmethod
    def prediction(self):
        pass

    @property
    @abstractmethod
    def loss(self):
        pass

    @property
    @abstractmethod
    def optimize(self):
        pass

    @property
    @abstractmethod
    def error(self):
        pass
