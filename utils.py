import os
import re

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


def __get_result_file_names():
    # get max index
    results_path = os.path.join(os.path.abspath('.'), 'results')
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    index = 0
    for x in os.listdir(results_path):
        matched = re.match(r'^run-(\d{4})$', x)
        if matched:
            cur = int(matched.group(1))
            if cur > index:
                index = cur
    index += 1

    # prepare dirs
    log_dir = 'results/run-{0:0>4d}/logs'.format(index)
    model_file_name = 'results/run-{0:0>4d}/model.h5'.format(index)
    return log_dir, model_file_name


def generate_callbacks_include_early_stop():
    log_dir, model_file_name = __get_result_file_names()

    return [
        # log
        TensorBoard(log_dir=log_dir),
        # save model if necessary
        ModelCheckpoint(filepath=model_file_name,
                        monitor='val_loss',
                        save_best_only=True),
        # early stop if acc is not improvement
        EarlyStopping(monitor='acc', patience=3),
        # early stop if val_loss is not improvement
        EarlyStopping(monitor='val_loss', patience=3)
    ]


def generate_callbacks():
    log_dir, model_file_name = __get_result_file_names()

    return [
        # log
        TensorBoard(log_dir=log_dir),
        # save model if necessary
        ModelCheckpoint(filepath=model_file_name,
                        monitor='val_loss',
                        save_best_only=True)
    ]