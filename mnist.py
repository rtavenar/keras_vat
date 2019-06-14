from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist

from model import VATModel, SemiSupervisedVATModel

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

n_filters = 5
kernel_size = 5
pool_size = 2
n_classes = 10

model = Sequential([
    Convolution2D(n_filters,
                  kernel_size,
                  activation='relu',
                  input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation="relu"),
    Dense(n_classes, activation='softmax')
])

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')[:, :, :, None]
X_test = X_test.astype('float32')[:, :, :, None]
X_train /= 255.
X_test /= 255.

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

# Fully supervised training: do not use VATModel
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=128, epochs=3, verbose=2)

# Fully unsupervised training: use VATModel without any other loss function
model2 = VATModel(inputs=model.inputs, outputs=model.outputs)
model2.compile(optimizer="adam", loss=[None] * len(model2.outputs),
               metrics=["accuracy"])
model2.fit(X_train, None, batch_size=128, epochs=3, verbose=2)

# Semi-supervised training: define a model with two inputs/outputs/losses, one
# for supervised data, one for unsupervised data
model3 = SemiSupervisedVATModel(inputs=model.inputs, outputs=model.outputs)
model3.compile(optimizer="adam", loss=["categorical_crossentropy", None],
               metrics=["accuracy"])
model3.fit([X_train_sup, X_train_unsup],
           [y_train_sup, None],
           batch_size=128, epochs=3, verbose=2)