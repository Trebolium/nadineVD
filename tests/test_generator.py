from vocalDetector import network
from vocalDetector import utils


params = utils.get_parameters()
my_generator = network.data_generator('train', 100, params)

for i in range(10):
    x, y = my_generator.__next__()
    print(x.shape, y.shape)
