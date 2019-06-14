from functools import reduce
from keras.utils.generic_utils import to_list
from keras.engine.training import Model
from keras.layers import Input
import keras.backend as K
import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class VATModel(Model):
    # Widely inspired from
    # https://gist.github.com/mokemokechicken/2658d036c717a3ed07064aa79a59c82d
    def __init__(self, model, input_shape=(28, 28, 1), xi=1e-3, eps=1.0, ip=1):  # TODO: input_shape
        vat_inputs = [Input(shape=input_shape) for i in model.inputs]  # TODO
        vat_outputs = model(vat_inputs)
        super(VATModel, self).__init__(inputs=vat_inputs,
                                       outputs=vat_outputs)
        self._vat_loss = None
        self.setup_vat_loss(eps=eps, xi=xi, ip=ip)

    def setup_vat_loss(self, eps, xi, ip):
        self._vat_loss = self.vat_loss(eps, xi, ip)
        self.add_loss(self._vat_loss)
        return self

    def vat_loss(self, eps, xi, ip):
        normal_outputs = [K.stop_gradient(x) for x in to_list(self.outputs)]
        d_list = [K.random_normal(K.shape(x)) for x in self.inputs]

        for _ in range(ip):
            new_inputs = [x + self.normalize_vector(d)*xi
                          for (x, d) in zip(self.inputs, d_list)]
            new_outputs = to_list(self.call(new_inputs))
            klds = [K.sum(self.kld(normal, new))
                    for normal, new in zip(normal_outputs, new_outputs)]
            kld = reduce(lambda t, x: t+x, klds, 0)
            d_list = [K.stop_gradient(d) for d in K.gradients(kld, d_list)]

        new_inputs = [x + self.normalize_vector(d) * eps
                      for (x, d) in zip(self.inputs, d_list)]
        y_perturbations = to_list(self.call(new_inputs))
        klds = [K.mean(self.kld(normal, new))
                for normal, new in zip(normal_outputs, y_perturbations)]
        kld = reduce(lambda t, x: t + x, klds, 0)
        return kld

    @staticmethod
    def normalize_vector(x):
        z = K.sum(K.batch_flatten(K.square(x)), axis=1)
        while K.ndim(z) < K.ndim(x):
            z = K.expand_dims(z, axis=-1)
        return x / (K.sqrt(z) + K.epsilon())

    @staticmethod
    def kld(p, q):
        v = p * (K.log(p + K.epsilon()) - K.log(q + K.epsilon()))
        return K.sum(K.batch_flatten(v), axis=1, keepdims=True)


class SemiSupervisedVATModel(Model):
    def __init__(self, model, input_shape=(28, 28, 1), xi=1e-3, eps=1.0, ip=1):
        supervised_model = model
        unsupervised_model = VATModel(model, input_shape=input_shape,
                                      xi=xi, eps=eps, ip=ip)
        all_inputs = supervised_model.inputs + unsupervised_model.inputs
        all_outputs = supervised_model.outputs + unsupervised_model.outputs
        super(SemiSupervisedVATModel, self).__init__(inputs=all_inputs,
                                                     outputs=all_outputs)

        self.supervised_model = supervised_model
        self.unsupervised_model = unsupervised_model

    def compile(self, optimizer, loss, metrics=[]):
        self.supervised_model.compile(optimizer=optimizer,
                                      loss=loss,
                                      metrics=metrics)
        self.unsupervised_model.compile(optimizer=optimizer, loss=[None])

    @property
    def metrics_names(self):
        return ["sup_" + m for m in self.supervised_model.metrics_names] + \
               ["unsup_" + m for m in self.unsupervised_model.metrics_names]

    def fit(self, x=None, y=None, batch_size=32, epochs=1):
        # One item for supervised, one for unsupervised
        assert len(x) == len(y) == 2
        n_sup = x[0].shape[0]
        n_unsup = x[1].shape[0]
        n = max([n_sup, n_unsup])
        for e in range(epochs):
            n_batches = n // batch_size
            loss_sup = numpy.array([0.] * len(self.supervised_model.metrics_names))
            loss_unsup = numpy.array([0.] * len(self.supervised_model.metrics_names))
            for b in range(n_batches):
                indices_sup = numpy.random.randint(low=0, high=n_sup,
                                                   size=batch_size)
                indices_unsup = numpy.random.randint(low=0, high=n_unsup,
                                                     size=batch_size)
                x_sup_batch = x[0][indices_sup]
                y_sup_batch = y[0][indices_sup]
                x_unsup_batch = x[1][indices_unsup]
                loss_sup += self.supervised_model.train_on_batch(x_sup_batch,
                                                                 y_sup_batch)
                loss_unsup += self.unsupervised_model.train_on_batch(x_unsup_batch,
                                                                     None)
            loss_sup /= n_batches
            loss_unsup /= n_batches
            loss = numpy.stack((loss_sup, loss_unsup)).reshape((-1, ))
            s = "epoch {}/{}: ".format(e + 1, epochs)
            for i, name in enumerate(self.metrics_names):
                s += "{}: {} ".format(name, loss[i])
            print(s)
        return [-1., -1.]  # TODO

    def predict(self, x,
                batch_size=None,
                verbose=0,
                steps=None):
        return self.supervised_model.predict(x=x, batch_size=batch_size,
                                             verbose=verbose, steps=steps)

    def evaluate(self, x=None, y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None):
        return self.supervised_model.evaluate(x=x, y=y,
                                              sample_weight=sample_weight,
                                              batch_size=batch_size,
                                              verbose=verbose, steps=steps)