import tensorflow as tf

from abc import ABC, abstractmethod
from pylat.exceptions import InvalidArgumentError


class LayerConfig:
    """Class that holds the parameters common to any layer object.

    This class can be used to initialize every :obj:`BaseLayer` object
    using their :func:`load_config` method.

    Parameters
    ----------
    num_units : int, optional (default=None)
        Number of units that the layer will have.
    kernel_init : :obj:`callable`, optional (default=None)
        Function used to initialize the weights of the layer's cells.
    activation : :obj:`callable`, optional (default=None)
        Activation function of the layer.
    dropout_rate : int, optional (default=None)
        Dropout rate of the layer.
    """

    def __init__(self, num_units=None, kernel_init=None, activation=None,
                 dropout_rate=None):
        self.num_units = num_units
        self.kernel_init = kernel_init
        self.activation = activation
        self.dropout_rate = dropout_rate

    def get_params(self, deep=True):
        """Return the parameters of the layer config in a dict."""
        return {
            "num_units": self.num_units,
            "kernel_init": self.kernel_init,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate
        }


class BaseLayer(ABC):
    """Abstract base class of all the layer creation objects.

    This class serves as a base of all the classes that create TensorFlow
    layers. It contains the parameters which are common to all of them, and
    defines a :func:`build_tensor` method that must be overriden to create the
    actual TensorFlow layer.

    Parameters
    ----------
    num_units : int
        Number of units that the layer will have.
    kernel_init : :obj:`callable`, optional (default='glorot_uniform')
        Function used to initialize the weights of the layer's cells.
    activation : :obj:`callable`, optional (default='relu')
        Activation function of the layer.
    dropout_rate : int, optional (default=None)
        Dropout rate of the layer.
    """

    def __init__(self, num_units, kernel_init='glorot_uniform',
                 activation='relu', dropout_rate=None):
        self.num_units = num_units
        self.kernel_init = kernel_init
        self.activation = activation
        self.dropout_rate = dropout_rate
        self._check_valid_params()

    def load_config(self, layer_config):
        """Load the parameters of the layer using a LayerConfig object.

        Parameters
        ----------
        layer_config : :obj:`LayerConfig`
            LayerConfig object that holds the parameters that will be loaded
            by the layer.
        Returns
        -------
        self
            Self reference after the parameters have been initialized.
        Raises
        ------
        InvalidArgumentError
            If the layer_config parameter is not an instance of the
            :obj:`LayerConfig` class.
        """
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
        """Build the TensorFlow tensor of the layer.

        This method must be overridden by specific BaseLayer classes to build
        and return their layer tensor.

        Parameters
        ----------
        inputs : :obj:`tf.Tensor`
            Input tensor of the layer.
        kwargs : dict
            Additional arguments used to create the layer.

        Returns
        -------
        tf.Tensor
            TensorFlow tensor of the created layer.
        """
        pass

    def _check_valid_params(self):
        if self.dropout_rate is None:
            return
        elif self.dropout_rate < 0 or self.dropout_rate > 1:
            raise InvalidArgumentError('dropout_rate', 'Dropout rate must be '
                                       'a float between 0 and 1.')


class DenseLayer(BaseLayer):
    """Class that creates dense TensorFlow layers."""

    def build_tensor(self, inputs, **kwargs):
        output = tf.keras.layers.Dense(self.num_units,
                                     kernel_initializer=self.kernel_init,
                                     activation=self.activation,
                                     name='dense_layer')(inputs)
        if self.dropout_rate is not None:
            output = tf.keras.layers.Dropout(self.dropout_rate)(output,
                                                                **kwargs)
        return output
