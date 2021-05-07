import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

class PositionalEncoder:
    """Compute positional angles between the elements
     to encode their positions in a sequence (TSP)"""

    def __init__(self, position, dim=20):
        self.position = position  # max lenth of a sequence
        self.dim = dim            # dimensionality of a model

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