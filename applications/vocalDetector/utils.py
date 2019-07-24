import csv
import os
import yaml
import numpy as np
from .features import mel_spectrogram, spectrogram


""" Some utility methods. """


def get_parameters():
    params = yaml.load(open('../parameters.yaml', 'r'))
    return params


def get_file_list(mode, params):
    root = params['data_folder']
    switcher = {
        'train': root + "jam_train_list.txt",
        'test': root + "jam_test_list.txt",
        'valid': root + "jam_valid_list.txt",
    }
    file_name = switcher.get(mode, 'not found')
    file_list = []
    if file_name == 'not found':
        return file_list
    with (open(file_name, 'r')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        for row in csv_reader:
            file_list.append(os.path.join(root, mode, row[0][:-4]))
    return file_list


def get_all_file_lists(params):
    datasets = ['train', 'test', 'valid']
    file_list = []
    for dataset in datasets:
        file_list.extend(get_file_list(dataset, params))
    return file_list


def save_spectrogram(file_name, spec):
    np.save(file_name + '.npy', spec)
    return


def save_stats(mel_spectra, params):
    mel_spectra = np.concatenate(mel_spectra, axis=1)
    m = np.mean(mel_spectra, axis=1)
    st = np.std(mel_spectra, axis=1)
    np.save(params['stats_folder'] + 'mean_vd_cnn', m)
    np.save(params['stats_folder'] + 'std_vd_cnn', st)
    return


def load_all_spectrograms(mode, params):
    """ Read all training spectrograms and return as array. """
    file_list = get_file_list(mode, params)
    all_specs = []
    for file_name in file_list:
        all_specs.append(load_spectrogram(file_name))
    return all_specs


def load_spectrogram(file_name):
    """ Load a spectrogram for a given file. """
    spec = np.load(file_name + '.npy')
    return spec


def get_label_for_frame_index(file_name, frame_index, params):
    """ Get the label for a given file and frame index. """
    annotation_file = params['annotation_folder'] + os.path.basename(file_name) + '.lab'
    frame_time = frame_index * params['fft_hop'] / float(params['fs'])
    prev_value = 0.0
    with open(annotation_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            start_time = float(row[0])
            if start_time > frame_time:
                return prev_value
            else:
                prev_value = float(row[2] == 'sing')
    return prev_value


def load_validation_data(params):
    """ Load validation data. """
    val_list = get_file_list('valid', params)
    #instances_per_song = int(np.ceil(params['num_val_instances'] / len(val_list)))
    x_val = []
    y_val = []
    m, s = load_statistics()
    for f, file_name in enumerate(val_list):
        spec = load_spectrogram(file_name)
        for i in range(params['context'], spec.shape[1] - params['context'] - 1, params['eval_hop']):
            frame_index = np.random.randint(params['context'], spec.shape[1] - params['context'] - 1)
            x_data = spec[:, frame_index - params['context']:frame_index + params['context'] + 1]
            x_data = mel_spectrogram(x_data, params)
            x_data = x_data.T - m
            x_data = x_data / s
            x_val.append(x_data)
            y_val.append(get_label_for_frame_index(file_name, frame_index, params))
    x_val = np.asarray(x_val)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    y_val = np.asarray(y_val)
    return x_val, y_val


def load_test_data(file_name, params):
    #mel = load_spectrogram(file_name)
    mel = spectrogram(file_name, params)
    m, s = load_statistics()
    x_data = []
    y = []
    for i in range(params['context'], mel.shape[1] - params['context'] - 1, params['eval_hop']):
        x = mel[:, i - params['context']:i + params['context'] + 1]
        x = mel_spectrogram(x, params)
        x = x.T - m
        x = x / s
        x_data.append(x)
        y.append(get_label_for_frame_index(file_name, i, params))
    x_data = np.asarray(x_data)
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)
    return x_data, y


def load_statistics():
    s = np.load('../stats/std_vd_cnn.npy')
    m = np.load('../stats/mean_vd_cnn.npy')
    return m, s
