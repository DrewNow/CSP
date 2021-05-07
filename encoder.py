""" Encoder: 

1/ Preprocess data:
- Embedding: composition of N atoms - > N x D matrix,
where D - model dimension (atom2vec dimension)
- Pad compositions with 0 if necessary
- Create padding and look-ahead masks
- Positional encoding - None: Bag of Atoms
                      - Try ordered by Pettifor

2/ M layers of EncoderLayer 

Pass output to Decoder
"""
import tensorflow as tf
from preprocessor import Preprocessor
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()
       
        self.d_model = d_model
        self.num_layers = num_layers
       
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)
       
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
       
        self.dropout = tf.keras.layers.Dropout(rate)
       
    def call(self, x, training, mask):
   
         seq_len = tf.shape(x)[1]
       
         # adding embedding and position encoding.
         x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
         x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
         x += self.pos_encoding[:, :seq_len, :]
       
         x = self.dropout(x, training=training)
       
         for i in range(self.num_layers):
           x = self.enc_layers[i](x, training, mask)
       
         return x  # (batch_size, input_seq_len, d_model)
