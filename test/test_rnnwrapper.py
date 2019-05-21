__author__ = 'Alejandro González Hevia'


# def test_save_restore(self):
#     rnn_layers = [RecurrentLayer(num_units=3, cell_factory=GRUCellFactory())]
#     fc_layers = [DenseLayer(num_units=5)]
#     model = RecurrentNeuralNetwork(rnn_layers, fc_layers,
#                                    embeddings=self.embeddings)
#     trainer = BaseTrainer(model, num_epochs=100, batch_size=1)
#     trainer.train(self.x_train, self.y_train)
#     model.save('tmp')
#     new_model = RecurrentNeuralNetwork(rnn_layers, fc_layers,
#                                        embeddings=self.embeddings)
#     new_model.restore('tmp')
#     assert old_prediction == new_model.prediction(self.x_val)
#     os.remove('tmp')
