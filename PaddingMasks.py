import tensorflow as tf 

class PaddingMasks:
    """ Prepare data for training in Transformer model """

    def __init__(self, data):
        self.data = data

    def call(self):
        padding_mask = self.create_padding_mask()
        look_ahead_mask = self.create_look_ahead_mask()

        return padding_mask, look_ahead_mask

    def create_padding_mask(self):
        """ Create tensor with ones at the padded positions in data """

        seq = tf.cast(tf.math.equal(self.data, 0), tf.float32)
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, phase_size)

    def create_look_ahead_mask(self):
        """ Mask future tokens in a sequence """
        size = self.data.shape[1]
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)