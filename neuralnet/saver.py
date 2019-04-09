from .model import ModelDecorator


class Saver(ModelDecorator):
    def __init__(self, model):
        super(ModelDecorator, self).__init__(self, model)

    def init_model(self, X, y):
        pass

    def on_epoch_finished(self, loss, acc):
        pass

    def on_train_finished(self):
        self.logger.info('Restoring checkpoint of best model...')
        self.saver.restore(self.session, self.save_path)

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