import numpy as np
import tensorflow as tf

from os import path
from PIL import Image


DATA_NAMES = ('x_train', 'p_train', 'y_train', 'x_test', 'p_test', 'y_test')
DATA_DIR = path.join(path.dirname(__file__), 'digits')


def _load_perms():
    ''' Loads permutations of digits '''
    data = []
    for digit in range(10):
        perm_list = []
        for perm in range(4):
            img_fpath = path.join(DATA_DIR, f'{digit}_{perm}.png')
            img = np.asarray(Image.open(img_fpath).convert('L').resize((28,28))).reshape((1,28,28,1))
            perm_list.append(img)
        perms = np.concatenate(perm_list, 0)
        data.append(perms)
    return data


def _generate_data(x, y):
    '''Generates MNIST dataset used for GRU_CNN model '''
    digits = list(range(10))
    digit_data = []
    new_perm_data = []
    new_perm_labels = []
    
    # Y labels
    perm_data = _load_perms() # List[np.ndarray(4,28,28,1)]
    
    for dgx in digits:
        # Load MNIST data into array
        digit_arr = x[y == dgx]
        n_data = int(digit_arr.shape[0] / 5)
        rem = digit_arr.shape[0] % 5
        if rem == 0:
            digit_arr = digit_arr.reshape((n_data,5,28,28,1))
        else:
            digit_arr = digit_arr[0:-rem].reshape((n_data,5,28,28,1))
        digit_data.append(digit_arr)
        
        # Load permutations
        perm_dgx = perm_data[dgx]
        perm_labels = np.array(range(4))
        new_perm = np.tile(perm_dgx, (int(n_data/4),1,1,1))
        new_labels = np.tile(perm_labels, int(n_data/4))
        rem = n_data % 4
        if rem != 0:
            end_perm = perm_dgx[0:rem,:,:,:]
            end_labels = perm_labels[0:rem]
            new_perm = np.concatenate([new_perm, end_perm], 0)
            new_labels = np.concatenate([new_labels, end_labels])
        new_perm_data.append(new_perm)
        new_perm_labels.append(new_labels)

    x_digit_out = np.concatenate(digit_data, 0) # (N,5,28,28,1)
    x_labels_out = np.concatenate(new_perm_labels) # (N,)
    y_perm_out = np.concatenate(new_perm_data, 0) # (N,28,28,1)
    
    return x_digit_out, x_labels_out, y_perm_out


def generate_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, p_train, y_train = _generate_data(x_train, y_train)
    x_test, p_test, y_test = _generate_data(x_test, y_test)

    return (x_train, p_train, y_train, x_test, p_test, y_test)


def save_data(data_tuple):
    np.savez(path.join(DATA_DIR, 'mnist_data.npz'), **dict(zip(DATA_NAMES, data_tuple)))


def load_data():
    # dataset = []
    npzfile = np.load(path.join(DATA_DIR, 'mnist_data.npz'))
    dataset = [npzfile[x] for x in DATA_NAMES]

    # Normalize data
    x_train, p_train, y_train, x_test, p_test, y_test = tuple(dataset)

    x_train, y_train = x_train / 255.0, y_train / 255.0
    x_test, y_test = x_test / 255.0, y_test / 255.0

    return (x_train, p_train, y_train, x_test, p_test, y_test)

def load_training_data(batch_size=256, buffer_size=10000):
    # Load data
    x_train, p_train, y_train, _, _, _ = load_data()

    # Convert p data to onehot representation
    p_train_hot = tf.one_hot(p_train, 4, on_value=None, off_value=None)

    train_data = tf.data.Dataset.from_tensor_slices(((x_train,p_train_hot),(y_train)))
    train_data = train_data.shuffle(buffer_size=buffer_size).batch(batch_size=batch_size).prefetch(1)

    return train_data


def generate_example_data():
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    '''Generates MNIST dataset used for GRU_CNN model '''
    digits = list(range(10))
    digit_data = []
    
    for dgx in digits:
        # Load MNIST data into array
        digit_arr = x_test[y_test == dgx]
        digit_arr = digit_arr[0:300]
        digit_out = digit_arr.reshape((1,300,28,28,1))
        digit_data.append(digit_out)

    x_digit_out = np.concatenate(digit_data, 0) # (10,300,28,28,1)
    
    return x_digit_out