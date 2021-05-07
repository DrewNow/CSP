""" Decoder:
1/ Output embedding (atom2vec embedding)
2/ Look-ahead mask
3/ Positional encoding (TSP)
4/ M layers of DecoderLayers

"""

import tensorflow as tf
from Embedding import Embedding
from PaddingMasks import PaddingMasks
from PositionalEncoder import PositionalEncoder
from DecoderLayer import DecoderLayer

class Decoder(tf.keras.layers.Layer):
    def __init__(self, data, num_layers, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()
      
        self.num_layers = num_layers

        self.embed = Embedding(data)
        self.maxlength = self.embed.maxl
        self.d_model = self.embed.d_model
        
        self.pos_encoding = PositionalEncoder(self.maxlength, self.d_model)
      
        self.dec_layers = [DecoderLayer(self.d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
   
    def call(self, x, enc_output, training):
   
        attention_weights = {}
     
        x = self.embed(x) # (batch_size, target_seq_len, d_model)

        padding_mask, look_ahead_mask = PaddingMasks(x)()

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :self.maximum_position_encoding, :]
        x = self.dropout(x, training=training)
     
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 self.look_ahead_mask, self.padding_mask)
     
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
     
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights