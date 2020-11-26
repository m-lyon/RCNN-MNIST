
import numpy as np

from os import path

def _load_digit_array():
    arr = np.load(path.join(path.dirname(__file__), 'data', 'digits.npy'))
    return arr / 255.0

class Tutorial:

    digit_arr = _load_digit_array()
    
    @classmethod
    def get_digit_set(cls, digit, num):
        '''Gets a random subset set of length `num` of digit `digit`

        Args:
            digit (int): digit from 0-9
            num (int): Number of returned digits

        Returns:
            digits (np.ndarray): Digit image set (1, num, 28, 28, 1)
        '''
        if not isinstance(num, int):
            raise AttributeError('num is not type integer.')
        if num < 0 or num > 300:
            raise AttributeError('num must be between 0 and 300.')
        if not isinstance(digit, int):
            raise AttributeError('digit is not type integer.')
        if digit < 0 or digit > 9:
            raise AttributeError('digit must be between 0 and 9.')
        sample = np.random.random_integers(0, 300, size=(num,))
        return cls.digit_arr[digit:digit+1,sample,:,:,:]