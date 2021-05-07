""" Encoder: 

1/ Preprocess data:
- Embedding: composition of N atoms - > N x D matrix,
where D - model dimension (atom2vec dimension)
- Pad compositions with 0 if necessary
- Create padding mask (encoder)
  and look-ahead masks (decoder)
- Positional encoding - None: Bag of Atoms (for encoder; alt: ordered by Pettifor)
                      - TSP (decoder)

2/ M layers of EncoderLayer 

Pass output to Decoder
"""
import tensorflow as tf
from Embedding import Embedding
from PaddingMasks import PaddingMasks
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer

class Encoder(tf.keras.layers.Layer):
    def __init__(self, data, num_layers, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
       
        self.embed = Embedding(data)
       
        self.enc_layers = [EncoderLayer(self.embed.d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
       
        self.dropout = tf.keras.layers.Dropout(rate)
       
    def call(self, x, training):

        x = self.embed(x)

        padding_mask, _ = PaddingMasks(x)()
   
        x = self.dropout(x, training=training)
      
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, padding_mask)
      
        return x  # (batch_size, input_seq_len, d_model)