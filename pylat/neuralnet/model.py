import os
import tensorflow as tf

from abc import ABC, abstractmethod


class BaseNeuralNetwork(ABC):
    """Base class of neural networks.

    This class provides the main interface and parameters common to any
    implementation of a neural network using TensorFlow as the backend.

    Parameters
    ----------
    save_dir : str, optional (default='results')
        Directory where the neural network will be saved.

    Attributes
    ----------
    x_t : :obj:`tf.Tensor`
        Tensor that holds the input words used when a train or prediction
        method is called.

    y_t : :obj:`tf.Tensor`
        Tensor that holds the input labels used when the train method is called.

    session : tf.Session
        TensorFlow session where the neural network operations will
        be executed.
    """
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        self.save_path = os.path.join(self.save_dir, 'best.ckpt')
        self.session = None
        self.x_t = None
        self.y_t = None
        self._logits = None
        self.init = None

    def init_model(self, X, y):
        """Initializes the neural network with the given input.

        This method creates the session and the Tensors of the neural network
        given the input X and y data.

        Parameters
        ----------
        X : :obj:`np.array`
            Two dimensional array containing the training instances.
        y : list
            List containing the labels of each instance.
        """
        self.session = tf.Session()
        self.prediction
        self.optimize
        self.error
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)
        self.additional_inits()

    def additional_inits(self):
        """Perform additional initialization steps in a neural network.

        This method can be overridden by neural network implementations to
        perform additional implementation steps that are not covered in the
        default initialization of the :obj:`BaseNeuralNetwork` class.
        """
        pass

    @abstractmethod
    def save(self, save_path):
        """Saves the neural network to the given path.

        Parameters
        ----------
        save_path : str
            Directory where the neural network will be saved.
        """
        pass

    @abstractmethod
    def restore(self, save_path):
        """Restores the neural network from the given path.

        The tensors and their weights will be recovered from the saved files
        in the directory.

        Parameters
        ----------
        save_path : str
            Directory that holds the saved data of the neural network to be
            restored.
        """
        pass

    @property
    @abstractmethod
    def prediction(self):
        """Return the prediction tensor of the neural network.

        This property must be implemented by each neural network implementation
        to return the TensorFlow operation that calculates the final prediction
        of the neural network.

        Returns
        -------
        tf.Tensor
        """
        pass

    @property
    @abstractmethod
    def loss(self):
        """Return the loss tensor of the neural network.

        This property must be implemented by each neural network implementation
        to return the TensorFlow operation that calculates the loss of the
        neural network in the training phase.

        Returns
        -------
        tf.Tensor
        """
        pass

    @property
    @abstractmethod
    def optimize(self):
        """Return the optimization tensor of the neural network.

        This property must be implemented by each neural network implementation
        to return the TensorFlow operation that performs the optimization steps
        of the network in the training phase.

        Returns
        -------
        tf.Tensor
        """
        pass

    @property
    @abstractmethod
    def error(self):
        """Return the error tensor of the neural network.

        This property must be implemented by each neural network implementation
        to return the TensorFlow operation that calculates the error of the
        network given some labelled data.

        Returns
        -------
        tf.Tensor
        """
        pass
