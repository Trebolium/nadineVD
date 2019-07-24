import librosa
import numpy as np
import random
from scipy.ndimage import affine_transform


""" Methods related to feature extraction. """


def read_audio(file_name, params):
    y, _ = librosa.audio.load(file_name + '.wav', sr=params['fs'], mono=True)
    return y


def spectrogram(file_name, params):
    y = read_audio(file_name, params)
    y = y / np.max(np.abs(y))
    spec = librosa.core.stft(y, n_fft=params['fft_len'], hop_length=params['fft_hop'])
    spec, _ = librosa.magphase(spec)
    return spec


def mel_spectrogram(spec, params):
    spec = np.maximum(spec, 1e-7)
    mel = librosa.feature.melspectrogram(S=spec,
                                         sr=params['fs'],
                                         n_mels=params['num_mels'],
                                         fmin=params['min_freq'],
                                         fmax=params['max_freq'])
    mel = librosa.amplitude_to_db(mel, ref=1.0)
    return mel


def shift_stretch(s, params):
    output = np.zeros(s.shape)
    stretch = 100 + random.randint(-params['pitch_shift_amount'], params['pitch_shift_amount'])
    stretch = float(stretch) / 100.
    shift = 100 + random.randint(-params['time_stretch_amount'], params['time_stretch_amount'])
    shift = float(shift) / 100.
    offset = (.5 * (s.shape[0] - s.shape[0] / stretch), 0)
    affine_transform(s, (1 / stretch, 1 / shift), output=output, mode='constant', cval=0, prefilter=True, order=3, offset=offset)
    return output


def filtering(spec, params):
    # logspace between 150Hz and 8kHz
    mu_space = np.geomspace(50, 8000, 100)
    mu_ind = random.randint(0, mu_space.shape[0]-1)
    mu = mu_space[mu_ind]
    freqs = librosa.fft_frequencies(sr=params['fs'], n_fft=params['fft_len'])
    mu_freq_ind = np.argmin(np.abs(freqs-mu))
    sigma = random.randint(50, 70) / 10.
    sigma_freq = np.abs(freqs[mu_freq_ind] - freqs[mu_freq_ind] * np.power(2.0, sigma * 100 / 1200))
    db = float(random.randint(-10, 10))
    amp = librosa.db_to_amplitude(db)
    filt = gaussian_filt(freqs, freqs[mu_freq_ind], sigma_freq)
    filt *= amp
    filt += 1.0
    return spec * filt


def augment(spec, params):
    spec = spec.T
    spec = filtering(spec, params)
    spec = shift_stretch(spec, params)
    return spec.T


def gaussian_filt(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
