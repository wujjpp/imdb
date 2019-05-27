from keras.datasets import imdb
from keras import models, layers, losses, optimizers, activations
import numpy as np
import utils
from monitor import monitor
import logging


@monitor()
def train():
    (train_data, train_labels), (test_data,
                                 test_labels) = imdb.load_data(num_words=10000)

    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    model = models.Sequential()
    model.add(
        layers.Dense(16, activation=activations.relu, input_shape=(10000, )))
    model.add(layers.Dense(16, activation=activations.relu))
    model.add(layers.Dense(1, activation=activations.sigmoid))

    model.summary()

    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.binary_crossentropy,
                  metrics=['acc'])

    callbacks = utils.generate_callbacks_include_early_stop()

    model.fit(x_train,
              y_train,
              batch_size=256,
              epochs=100,
              validation_split=0.5,
              callbacks=callbacks)

    result = model.evaluate(x_test, y_test)
    print('evaluate result on 25000 samples:', result)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)    
    train()
