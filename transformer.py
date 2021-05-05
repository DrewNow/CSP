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


class Prepocessor:
    """ Prepare data for training in Transformer model """

    def __init__(self, data):
        self.data = data












    @staticmethod
    def create_depettifor(pettifor):
        """ Create Pettifor -> Atomic symbols dictionary """
        return dict(map(reversed, pettifor.items()))

    @staticmethod
    def tokenize(phase):
        """ Represent phases of elements as lists of Pettifor numbers """
        return np.array([pettifor[element] for element in phase])

    @staticmethod
    def detokenize(phase):
        """ Invert tokenization to Pettifor -> Atomic symbol """
        return [depettifor[element] for element in phase]

    @staticmethod
    def pad(phase, maxl=8):
        """ Add zeros at the end of phase vector to equalize lengths """
        if len(phase) == maxl:
            return phase
        else:
            padded = np.zeros(maxl)
            padded[:len(phase)] = phase
            return padded

    @staticmethod
    def create_padding_mask(seq):
        """ Create tensor with ones at the padded positions in data """

        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


class PositionalEncoder:
    """Compute positional angles between the elements
     to encode their positions in a sequence (TSP)"""

    def __init__(self, position, dim=len(pettifor)):
        self.position = position
        self.dim = dim  # dimensionality of a model, e.g. len(pettifor)
        self.angles = self.encode()
        self.angles = self.tfcast(self.angles)

    def get_angles(self, pos, i):
        angle = 1 / np.power(10000, (2 * (i//2)) / np.float(self.dim))
        return pos * angle

    def encode(self):
        angles = self.get_angles(np.arange(self.position)[:, np.newaxis],
                                 np.arange(self.dim)[:, np.newaxis])
        
        # apply sin to even indices in the array; 2i
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        return angles[np.newaxis, ...]
   
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

    @staticmethod
    def tfcast(encoding):
        return tf.cast(encoding, dtype=tf.float32)
