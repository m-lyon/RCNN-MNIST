import numpy as np
import tensorflow as tf

from os import path
from tensorflow import keras
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GRU, Conv2DTranspose, Concatenate, Reshape, UpSampling2D


class RCNNModel:
    '''Wrapper class to interact with GRU CNN model
    
    Args:
        load_weights (bool): Load pretrained model weights.
    '''
    def __init__(self, load_weights=True):
        self._load_weights = load_weights
        self.state = None
        self.encoder, self.decoder, self.autoencoder = gru_cnn_model()
        if self._load_weights:
            self.load_weights()

    def load_weights(self):
        self.autoencoder.load_weights(path.join(path.dirname(__file__), 'data', 'weights.h5'))

    def encode(self, tensor):
        try:
            if tensor.ndim != 5:
                raise RuntimeError('tensor must be 5 dimensional with shape (1, N, 28, 28, 1)')
            x, _, h, w, c = tensor.shape
            if (x, h, w, c) != (1, 28, 28, 1):
                raise RuntimeError('tensor must have shape (1, N, 28, 28, 1)')
        except AttributeError:
            raise AttributeError('Input tensor must be tensor/numpy array')
        self.state = self.encoder.predict(tensor)

    def decode(self, tensor):
        try:
            if tensor.ndim != 2:
                raise RuntimeError('tensor must be 2 dimensional with shape (1, 4)')
            x, y = tensor.shape
            if (x, y) != (1, 4):
                raise RuntimeError('tensor must have shape (1, 4)')
        except AttributeError:
            raise AttributeError('Input tensor must be tensor/numpy array')
        return self.decoder((tensor, self.state))


def gru_cnn_model(time_steps=None):
    '''Genereates the DAG model for the GRU CNN autoencoder
    
    Args:
        time_steps (int): Number of timesteps in model, set as None to
            allow varying number of timesteps.

    Returns:
        encoder (tf.keras.models.Model): Encoder model that takes in time series
            of same digits
        decoder (tf.keras.models.Model): Decoder that uses state tensor from encoder,
            plus a rotation vector to output a rotated digit
        autoencoder (tf.keras.models.Model): Autoencoder used for training encoder &
            decoder models.
    '''
    
    # Example input
    input_img = keras.Input(shape=(time_steps, 28, 28, 1))
    
    # Encoder
    tensor = TimeDistributed(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))(input_img)
    tensor = TimeDistributed(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))(tensor)
    tensor = TimeDistributed(MaxPooling2D(2))(tensor)
    tensor = TimeDistributed(Dropout(0.25))(tensor)
    tensor = TimeDistributed(Flatten())(tensor)
    tensor = TimeDistributed(Dense(128, activation='relu'))(tensor)
    encoder_output = GRU(units=50)(tensor)
    # Create encoder model instance
    encoder = keras.models.Model(inputs=input_img, outputs=encoder_output, name='encoder')
    
    # Decoder
    input_rotation = keras.Input(shape=(4,))
    input_encoder = keras.Input(shape=(50,))
    tensor = Concatenate()([input_rotation, input_encoder])
    tensor = Dense(128, activation='relu')(tensor)
    tensor = Dense(9216)(tensor)
    tensor = Reshape((12, 12, 64))(tensor)
    tensor = UpSampling2D(2)(tensor)
    tensor = Conv2DTranspose(filters=32, kernel_size=(3,3), activation='relu')(tensor)
    decoder_output = Conv2DTranspose(filters=1, kernel_size=(3,3), activation='relu')(tensor)
    # Create decoder model instance
    decoder = keras.models.Model(inputs=[input_rotation, input_encoder], outputs=decoder_output, name='decoder')
    
    # Create auto-encoder
    autoencoder_img = keras.Input(shape=(time_steps, 28, 28, 1))
    autoencoder_rot = keras.Input(shape=(4,))
    encoded_img = encoder(autoencoder_img)
    xfmed_img = decoder([autoencoder_rot, encoded_img])
    autoencoder = keras.models.Model(inputs=[autoencoder_img, autoencoder_rot], outputs=xfmed_img, name='semi-autoencoder')
    
    return encoder, decoder, autoencoder
