from vocalDetector.utils import *
from vocalDetector.features import *

""" Extract spectrograms for all training, test and validation songs and save as numpy arrays. Compute mean and 
standard deviation over mel-spectra of the train set and save as numpy arrays."""

params = get_parameters()
file_list = get_all_file_lists(params)
all_mel_specs = []

for i, file_name in enumerate(file_list):
    print(i+1, '/', len(file_list), file_name)
    spec = spectrogram(file_name, params)
    save_spectrogram(file_name, spec)
    print(spec.shape)
    if '/train' in file_name:
        all_mel_specs.append(mel_spectrogram(spec, params))

print("computing training set statistics...")
save_stats(all_mel_specs, params)

print("All done!")
