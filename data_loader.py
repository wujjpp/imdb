import os
import numpy as np
from progress.bar import Bar
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json
import gc
from monitor import Monitor

# download data from:
base_dir = '/home/jp/workspace'
imbd_dir = os.path.join(base_dir, 'aclImdb')
train_dir = os.path.join(imbd_dir, 'train')
test_dir = os.path.join(imbd_dir, 'test')

one_hot_data_file_name = os.path.join(base_dir, 'imdb-one-hot.npz')
one_hot_tokenizer_json_file_name = os.path.join(base_dir,
                                                'imdb-one-hot-tokenizer.json')

embedding_data_file_name = os.path.join(base_dir, 'imdb-embedding.npz')
embedding_tokenizer_json_file_name = os.path.join(
    base_dir, 'imdb-embedding-tokenizer.json')

__one_hot_tokenizer = None
__embedding_tokenizer = None


def __load_original_data(text_dir):
    texts = []
    labels = []

    types = ['neg', 'pos']

    total_file = 0
    for label_type in types:
        dir_name = os.path.join(text_dir, label_type)
        total_file += len(os.listdir(dir_name))

    with Bar('Processing:{}'.format(text_dir), max=total_file) as bar:
        for label_type in types:
            dir_name = os.path.join(text_dir, label_type)

            for fname in os.listdir(dir_name):
                bar.next()
                if fname[-4:] == '.txt':
                    with open(os.path.join(dir_name, fname)) as f:
                        texts.append(f.read())
                    if label_type == 'neg':  # 消极的
                        labels.append(0.)
                    else:  # 积极的
                        labels.append(1.)

    texts = np.asarray(texts)
    labels = np.asarray(labels, dtype='float32')

    # 打乱训练数据
    indices = np.arange(texts.shape[0])
    np.random.shuffle(indices)
    texts = texts[indices]  # numpy具备的功能
    labels = labels[indices]

    return texts, labels


def load_one_hot_data(max_words=10000):

    if not os.path.exists(one_hot_data_file_name):
        tokenizer = Tokenizer(num_words=max_words)

        with Monitor('load data'):
            train_texts, y_train = __load_original_data(train_dir)
            test_texts, y_test = __load_original_data(test_dir)

        with Monitor('fit on texts'):
            tokenizer.fit_on_texts(train_texts)

        with Monitor('convert train texts to matrix'):
            x_train = tokenizer.texts_to_matrix(train_texts, mode='binary')
            del train_texts
            gc.collect()

        with Monitor('convert test texts to matrix'):
            x_test = tokenizer.texts_to_matrix(test_texts, mode='binary')
            del test_texts
            gc.collect()

        with Monitor('save data to data file'):
            np.savez(one_hot_data_file_name,
                     x_train=x_train,
                     y_train=y_train,
                     x_test=x_test,
                     y_test=y_test)

        with Monitor('save tokenizer json file'):
            with open(one_hot_tokenizer_json_file_name, 'w') as f:
                f.write(tokenizer.to_json())

    else:
        with Monitor('load data from data file'):
            data = np.load(one_hot_data_file_name)
            x_train = data['x_train']
            y_train = data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']

    return (x_train, y_train), (x_test, y_test)


def get_one_hot_tokenizer():
    global __one_hot_tokenizer
    if not __one_hot_tokenizer:
        with open(one_hot_tokenizer_json_file_name, 'r') as f:
            __one_hot_tokenizer = tokenizer_from_json(f.read())

    return __one_hot_tokenizer


def load_embedding_data(max_words=10000, maxlen=200):
    if not os.path.exists(embedding_data_file_name):
        tokenizer = Tokenizer(num_words=max_words)

        with Monitor('load data'):
            train_texts, y_train = __load_original_data(train_dir)
            test_texts, y_test = __load_original_data(test_dir)

        with Monitor('fit on texts'):
            tokenizer.fit_on_texts(train_texts)

        with Monitor('convert train texts to sequences'):
            x_train = tokenizer.texts_to_sequences(train_texts)

            del train_texts
            gc.collect()

        with Monitor('convert test texts to sequences'):
            x_test = tokenizer.texts_to_sequences(test_texts)

            del test_texts
            gc.collect()

        with Monitor('save data to data file'):
            np.savez(embedding_data_file_name,
                     x_train=x_train,
                     y_train=y_train,
                     x_test=x_test,
                     y_test=y_test)

        with Monitor('save tokenizer json file'):
            with open(embedding_tokenizer_json_file_name, 'w') as f:
                f.write(tokenizer.to_json())

    else:
        with Monitor('load data from data file'):
            data = np.load(embedding_data_file_name)
            x_train = data['x_train']
            y_train = data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']

    # pad sequences
    x_train = pad_sequences(x_train,
                            maxlen=maxlen,
                            padding='post',
                            truncating='post')
    x_test = pad_sequences(x_test,
                           maxlen=maxlen,
                           padding='post',
                           truncating='post')
    return (x_train, y_train), (x_test, y_test)


def get_embedding_tokenizer():
    global __embedding_tokenizer
    if not __embedding_tokenizer:
        with open(one_hot_tokenizer_json_file_name, 'r') as f:
            __embedding_tokenizer = tokenizer_from_json(f.read())

    return __embedding_tokenizer
