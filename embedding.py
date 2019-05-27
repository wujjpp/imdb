from keras import models, layers, losses, optimizers, activations
import utils
import data_loader
from monitor import monitor
import logging


@monitor()
def train():

    max_words = 10000
    maxlen = 500

    (x_train,
     y_train), (x_test,
                y_test) = data_loader.load_embedding_data(max_words=max_words,
                                                          maxlen=maxlen)

    model = models.Sequential()

    model.add(
        layers.Embedding(input_dim=max_words,
                         output_dim=8,
                         input_length=maxlen))

    model.add(layers.Flatten())

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
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG)
    train()
