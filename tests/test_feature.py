from vocalDetector.features import mel_spectrogram, spectrogram
from vocalDetector.utils import *
import matplotlib.pyplot as plt

params = get_parameters()
val_list = get_file_list('valid', params)
file_name = val_list[0]

spec = spectrogram(file_name, params)
mel = mel_spectrogram(spec, params)

plt.imshow(mel, aspect='auto', origin='lower')
plt.colorbar()
plt.show()


