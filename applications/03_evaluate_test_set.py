from keras.models import load_model
from vocalDetector.utils import get_parameters, load_test_data, get_file_list
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

params = get_parameters()
model = load_model('../models/vocal_detector.h5')

all_accuracies = []
test_list = get_file_list('test', params)
all_true = []
all_est = []
for f, file_name in enumerate(test_list):
    print(f + 1, '/', len(test_list), file_name)
    print('loading test data...')
    x_test, y_test = load_test_data(file_name, params)
    print('predicting...')
    y_est = model.predict(x_test, verbose=1)
    print('evaluating...')
    acc = accuracy_score(y_test, np.round(y_est))
    print('accuracy: ', acc)
    print('bl: ', sum(y_test) / float(len(y_test)))
    all_accuracies.append(acc)
    all_true.extend(y_test)
    all_est.extend(y_est)
    #plt.plot(y_test)
    #plt.plot(y_est)
    #plt.show()

print(' =============================== ')
print('mean accuracy: ', np.mean(all_accuracies))
print('frame-wise accuracy: ', accuracy_score(np.round(all_true), np.round(all_est)))
