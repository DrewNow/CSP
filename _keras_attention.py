import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention


def attention(inputs):
    layer = MultiHeadAttention(num_heads=8, key_dim=2, value_dim=3, output_shape=(5,))
    target = tf.keras.Input(shape=[1000,23])
    source = tf.keras.Input(shape=[1000,23])
    output_tensor = layer(target, source)

if __name__ == '__main__':

#    layer = MultiHeadAttention(num_heads=8, key_dim=2, value_dim=2)#, attention_axes=(0))
    layer = MultiHeadAttention(num_heads=8, key_dim=2, value_dim=2, attention_axes=(0,1))
    input_tensor = tf.keras.Input(shape=(8,87,))
    output_tensor = layer(input_tensor, input_tensor)

    print(output_tensor.shape)
