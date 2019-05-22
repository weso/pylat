import tensorflow as tf

from abc import ABC, abstractmethod
from pylat.exceptions import InvalidArgumentError


class LayerConfig():
    def __init__(self, num_units=None, kernel_init=None, activation=None,
                 dropout_rate=None):
        self.num_units = num_units
        self.kernel_init = kernel_init
        self.activation = activation
        self.dropout_rate = dropout_rate

    def get_params(self, deep=True):
        return {
            "num_units": self.num_units,
            "kernel_init": self.kernel_init,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate
        }


class BaseLayer(ABC):
    def __init__(self, num_units, kernel_init='glorot_uniform',
                 activation='relu', dropout_rate=None):
        self.num_units = num_units
        self.kernel_init = kernel_init
        self.activation = activation
        self.dropout_rate = dropout_rate
        self._check_valid_params()

    def load_config(self, layer_config):
        if not isinstance(layer_config, LayerConfig):
            error_msg = "Layer config must be an instance of the LayerConfig" \
                        "class."
            raise InvalidArgumentError(layer_config, error_msg)

        for parameter, value in layer_config.get_params().items():
            if value is not None:
                setattr(self, parameter, value)
        return self

    @abstractmethod
    def build_tensor(self, inputs, **kwargs):
        pass

    def _check_valid_params(self):
        if self.dropout_rate is None:
            return
        elif self.dropout_rate < 0 or self.dropout_rate > 1:
            raise InvalidArgumentError('dropout_rate', 'Dropout rate must be '
                                       'a float between 0 and 1.')


class DenseLayer(BaseLayer):
    def build_tensor(self, inputs, **kwargs):
        output = tf.keras.layers.Dense(self.num_units,
                                     kernel_initializer=self.kernel_init,
                                     activation=self.activation,
                                     name='dense_layer')(inputs)
        if self.dropout_rate is not None:
            output = tf.keras.layers.Dropout(self.dropout_rate)(output,
                                                                **kwargs)
        return output
