from functools import reduce
from keras.utils.generic_utils import to_list
from keras.engine.training import Model
import keras.backend as K

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class VATModel(Model):
    # Widely inspired from
    # https://gist.github.com/mokemokechicken/2658d036c717a3ed07064aa79a59c82d
    def __init__(self, inputs, outputs, xi=1e-3, eps=1.0, ip=1):
        super(VATModel, self).__init__(inputs=inputs, outputs=outputs)
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
    def __init__(self, inputs, outputs):
        super(SemiSupervisedVATModel, self).__init__(inputs=inputs,
                                                     outputs=outputs)
        # TODO