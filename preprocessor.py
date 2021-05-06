# Training a transformer to recognise sequences (decoder with positional encoding) 
# from a beg of elements (sets -> encoder). 
#
# Alternative: Sets can be ordered wrt Pettifor number

import pandas as pd
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
# pettifor dictionary to import
from pettifor import pettifor


class Preprocessor:
    """ Prepare data for training in Transformer model """

    def __init__(self, data):
        self.data = data

    def call(self):
        # atom2vec atoms in data and pad
        tokenized_data = np.array(list(map(self.tokenize_and_pad, self.data)))
        padding_mask = self.create_padding_mask(tokenized_data)
        positional_encoding = PositionalEncoder(tokenized_data)

        return tokenized_data, padding_mask, positional_encoding

    @staticmethod
    def tokenize_and_pad(phase, maxl=8):
        """ Represent phases of m elements as a matrix (m, n)
        with n-dimensional atom2vec representation """

        vector = np.array([atom2vec(element) for element in phase])
        length = len(phase)
        if length == maxl:
            return vector
        else:
            return np.concatenate((vector, np.zeros(maxl-length)), axis=0)

    @staticmethod
    def detokenize(phase):            
        pass

    @staticmethod
    def create_padding_mask(phase):
        """ Create tensor with ones at the padded positions in data """

        seq = tf.cast(tf.math.equal(phase, 0), tf.float32)
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, phase_size)

    @staticmethod
    def create_look_ahead_mask(size):
        """ Mask future tokens in a sequence """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
 
class PositionalEncoder:
    """Compute positional angles between the elements
     to encode their positions in a sequence (TSP)"""

    def __init__(self, position, dim=20):
        self.position = position
        self.dim = dim  # dimensionality of a model

    def get_angles(self, pos, i):
        angle = 1 / np.power(10000, (2 * (i//2)) / np.float(self.dim))
        return pos * angle

    def call(self):
        angles = self.get_angles(np.arange(self.position)[:, np.newaxis],
                                 np.arange(self.dim)[:, np.newaxis])
        
        # apply sin to even indices in the array; 2i
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        return tf.cast(angles[np.newaxis, ...], dtype=tf.float32)
   
    @staticmethod
    def plot_positional_encoding(angles):
        print("Positional encoding shape:", angles.shape)
        n, d, _ = angles.shape
        angles = angles.reshape(n, d//2, 2)
        angles = angles.T 
        angles = angles.reshape(d, n)

        plt.pcolormesh(angles, cmap='RdBu')
        plt.xlabel('Depth')
        plt.ylabel('Position')
        plt.colorbar()
        plt.show