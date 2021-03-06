import tensorflow as tf
from MHA import MultiHeadAttention

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
   
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
   
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
   
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        return tf.keras.Sequential([
               tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
               tf.keras.layers.Dense(d_model)                  # (batch_size, seq_len, d_model)
               ])
   
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)               # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)                # (batch_size, input_seq_len, d_model)
   
        ffn_output = self.ffn(out1)                            # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)              # (batch_size, input_seq_len, d_model)
   
        return out2

if __name__ == '__main__':
    sample_ffn = point_wise_feed_forward_network(512, 2048)
    print('sample_ffn:')
    print('input shape: (512, 2048) - (d features, n samples in data)')
    print('point_wise_ffw returns tensor of shape:', sample_ffn(tf.random.uniform((64, 50, 512))).shape)
    print('(batch_size, input_seq_len, d_model)')
    print()
    
    sample_encoder_layer = EncoderLayer(512, 8, 2048)
   
    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)
    print('EncoderLayer(512, 8, 2048) for (64, 43, 512)')
    print('(batch_size, input_seq_len, d_model)')
    print('returns tensor of shape:')
    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
