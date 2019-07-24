from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from .utils import load_validation_data, load_all_spectrograms, get_file_list, get_label_for_frame_index, load_statistics
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from .features import augment, mel_spectrogram


def generate_network(params):
    model = Sequential()
    model.add(Convolution2D(64, (3, 3),
                            activation='relu',
                            input_shape=(2 * params['context'] + 1, params['num_mels'], 1)))
    model.add(Dropout(0.4))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Flatten())
    model.add(Dense(265, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.001)
    metrics = ['acc']
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=metrics)
    model.summary()
    return model


def data_generator(mode, num_instances, params):
    specs = load_all_spectrograms(mode, params)
    train_list = get_file_list('train', params)
    m, s = load_statistics()
    for i in range(num_instances):
        x_data = []
        y_data = []
        for ii in range(params['batch_size']):
            song_index = np.random.randint(len(specs)-1)
            spec = specs[song_index]
            frame_index = np.random.randint(params['context'], spec.shape[1] - params['context'] - 1)
            x = spec[:, frame_index - params['context']:frame_index + params['context'] + 1]
            if mode == 'train':
                x = augment(x, params)
            x = mel_spectrogram(x, params)
            x = x.T - m
            x = x / s
            y = get_label_for_frame_index(train_list[song_index], frame_index, params)
            x_data.append(x)
            y_data.append(y)
        x_data = np.asarray(x_data)
        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)
        y_data = np.asarray(y_data)
        yield x_data, y_data


def train(params):

    print('loading model...')
    model = generate_network(params)
    saveBest = ModelCheckpoint('../models/' + 'vocal_detector.h5', monitor='val_loss', save_best_only=True)
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=0, mode='auto')

    print('loading validation data ...')
    x_val, y_val = load_validation_data(params)
    print(x_val.shape, y_val.shape)

    print('training...')
    num_train_steps = np.floor(params['num_train_instances'] / params['batch_size'])
    model.fit_generator(data_generator('train', params['num_train_instances'], params),
                        steps_per_epoch=num_train_steps,
                        epochs=params['num_epochs'],
                        validation_data=(x_val, y_val),
                        callbacks=[earlyStopping, saveBest])

    print('All done!')
    return
