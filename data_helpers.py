import numpy as np
import scipy.io


def one_hot_encode(a):
    res = np.zeros((a.size, a.max() + 1))
    res[np.arange(a.size), a] = 1
    return res


def load_svhn(path):
    train_data = scipy.io.loadmat(path + '/train_32x32.mat')
    test_data = scipy.io.loadmat(path + '/test_32x32.mat')
    x_train, y_train = train_data['X'], train_data['y']
    x_test, y_test = test_data['X'], test_data['y']

    X_train = np.transpose(x_train, [3, 0, 1, 2])
    X_test = np.transpose(x_test, [3, 0, 1, 2])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = y_train.reshape(-1, ) - 1
    Y_test = y_test.reshape(-1, ) - 1
    # convert class vectors to binary class matrices
    Y_train = one_hot_encode(Y_train)
    Y_test = one_hot_encode(Y_test)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_train, Y_train, X_test, Y_test


def load_shvn_for_svm(path):
    train_data = scipy.io.loadmat(path + '/train_32x32.mat')
    test_data = scipy.io.loadmat(path + '/test_32x32.mat')
    x_train, y_train = train_data['X'], train_data['y']
    x_test, y_test = test_data['X'], test_data['y']

    X_train = np.transpose(x_train, [3, 0, 1, 2])
    X_test = np.transpose(x_test, [3, 0, 1, 2])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = y_train.reshape(-1, ) - 1
    Y_test = y_test.reshape(-1, ) - 1
    # convert class vectors to binary class matrices
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_train, Y_train, X_test, Y_test


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
