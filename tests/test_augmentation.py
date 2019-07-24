from vocalDetector.features import mel_spectrogram, spectrogram, augment
from vocalDetector.utils import *
import matplotlib.pyplot as plt

params = get_parameters()
val_list = get_file_list('valid', params)
file_name = val_list[0]
frame_index = 100

spec = spectrogram(file_name, params)
x = spec[:, frame_index - params['context']:frame_index + params['context'] + 1]
x_augmented = augment(x, params)

original = mel_spectrogram(x, params)
augmented = mel_spectrogram(x_augmented, params)


plt.subplot(211)
plt.imshow(original, aspect='auto', origin='lower')
plt.colorbar()
plt.subplot(212)
plt.imshow(augmented, aspect='auto', origin='lower')
plt.colorbar()
plt.show()