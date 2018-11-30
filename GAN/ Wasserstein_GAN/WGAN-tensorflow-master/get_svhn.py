import sys
import os
from six.moves import urllib
from scipy.io import loadmat
import numpy as np

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def maybe_download(data_dir):
    new_data_dir = os.path.join(data_dir, 'svhn')
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', new_data_dir+'/train_32x32.mat', _progress)
        filepath, _ = urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', new_data_dir+'/test_32x32.mat', _progress)

def load(data_dir, subset='train'):
    maybe_download(data_dir)
    if subset=='train':
        train_data = loadmat(os.path.join(data_dir, 'svhn') + '/train_32x32.mat')
        trainx = train_data['X']
        trainy = train_data['y'].flatten()
        trainy[trainy==10] = 0
        trainx = trainx.transpose((3, 0, 1, 2))
        trainy = dense_to_one_hot(trainy, 10)
        return trainx, trainy
    elif subset=='test':
        test_data = loadmat(os.path.join(data_dir, 'svhn') + '/test_32x32.mat')
        testx = test_data['X']
        testy = test_data['y'].flatten()
        testy[testy==10] = 0
        testx = testx.transpose((3, 0, 1, 2))
        testy = dense_to_one_hot(testy, 10)
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')

def main():
    # maybe_download('./')
    tx, ty = load('./')
    print(tx.shape)


if __name__ == '__main__':
    main()