import tensorflow as tf
import math
import numpy as np


def count_elements(dataset):
    num_elements = 0
    for element in dataset:
        num_elements += element[0][0].shape[0]
    return num_elements


class LrFindCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, decay_epoch=10):
        self.x_test = x_test
        self.y_test = y_test
        self.decay_epoch = decay_epoch
        self.steps_per_epoch = 1

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            print("Finding optimal learning rate...")
            self.steps_per_epoch = self.model.optimizer._learning_rate.decay_steps // self.decay_epoch
            loss_weights = self.model.compiled_loss._loss_weights
            model = tf.keras.models.clone_model(self.model)
            lr_finder = LRFinder(model, loss=self.model.loss, loss_weights=loss_weights)
            lr_finder.find(self.x_test, data_y=self.y_test, start_lr=1e-5, end_lr=1e-1, epochs=5, steps_per_epoch=100)
            max_lr = lr_finder.get_best_lr(sma=50)
            self.model.optimizer._learning_rate.initial_learning_rate = max_lr

    def on_epoch_end(self, epoch, logs=None):
        logs["lr"] = self.model.optimizer.lr.numpy()
        return logs


class LRFinder:
    def __init__(self, model, loss="MSE", loss_weights=None):
        self.model = model
        self.losses = []
        self.learning_rates = []
        self.best_loss = 1e9
        if loss_weights is None:
            self.model.compile(loss=loss,
                               optimizer=tf.keras.optimizers.Adam(0.001, clipvalue=10.0))
        else:
            self.model.compile(loss=loss,
                               optimizer=tf.keras.optimizers.Adam(0.001, clipvalue=10.0),
                               loss_weights=loss_weights)
        self.best_lr = 0
        self.lr_mult = 0

    def on_batch_end(self, batch, logs):
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

        loss = logs['loss']
        self.losses.append(loss)

        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        lr *= self.lr_mult
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def find(self, data_x, start_lr, end_lr, data_y=None, epochs=1, steps_per_epoch=None, **kw_fit):
        if steps_per_epoch is None:
            raise Exception('To correctly train on the data generator,`steps_per_epoch` cannot be None.'
                            'You can calculate it as `np.ceil(len(TRAINING_LIST) / BATCH)`')
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(epochs * steps_per_epoch))
        initial_weights = self.model.get_weights()
        original_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, start_lr)
        callback = tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))
        if tf.distribute.get_strategy() in [tf.distribute.MirroredStrategy,
                                            tf.distribute.MultiWorkerMirroredStrategy]:
            if data_y is not None:
                elements = max(data_x[0].shape)
                data_x = tf.data.Dataset.from_tensor_slices((data_x, data_y))
                data_x = data_x.shuffle(buffer_size=elements)
                data_x = data_x.batch(math.ceil(elements / steps_per_epoch))
                options = tf.data.Options()
                options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
                data_x = data_x.with_options(options)
            else:
                elements = count_elements(data_x)
                data_x = data_x.unbatch().shuffle(elements).batch(math.ceil(elements / steps_per_epoch))
            self.model.fit(data_x, epochs=epochs, callbacks=[callback], **kw_fit, verbose=0)
        else:
            self.model.fit(x=data_x, y=data_y, epochs=epochs, callbacks=[callback], **kw_fit, verbose=0)
        self.model.set_weights(initial_weights)
        tf.keras.backend.set_value(self.model.optimizer.lr, original_lr)

    def get_learning_rates(self):
        return self.learning_rates

    def get_losses(self):
        return self.losses

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.learning_rates)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=5, n_skip_end=1):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        self.best_lr = self.learning_rates[n_skip_beginning:-n_skip_end][best_der_idx]
        return self.best_lr
