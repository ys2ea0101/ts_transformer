import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv1D


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, look_ahead_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1 = self.mha1(
            x, x, x, look_ahead_mask
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(
            ffn_output + out1
        )  # (batch_size, target_seq_len, d_model)

        return out3


class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, seq_length, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name="Time2VecLayer")
        self.seq_length = seq_length
        self.tau = tf.reshape(tf.range(seq_length), [1, seq_length, 1])
        self.tau = tf.cast(self.tau, tf.float32)
        self.k = kernel_size

    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(
            name="wb", shape=(1,), initializer="uniform", trainable=True
        )
        self.bb = self.add_weight(
            name="bb", shape=(1,), initializer="uniform", trainable=True
        )
        # periodic
        self.wa = self.add_weight(
            name="wa",
            shape=(1, self.k),
            initializer="uniform",
            trainable=True,
        )
        self.ba = self.add_weight(
            name="ba",
            shape=(1, 1, self.k),
            initializer="uniform",
            trainable=True,
        )
        super(Time2Vec, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        # (1, s, 1)
        bias = self.wb * self.tau + self.bb
        # (1, s, k)
        dp = K.dot(self.tau, self.wa) + self.ba
        wgts = K.sin(dp)  # or K.cos(.)

        # (1, s, k+1)
        ret = K.concatenate([bias, wgts], -1)
        return ret

    def compute_output_shape(self, input_shape):
        return (1, self.seq_length, self.k + 1)


class time_delayed_embed(tf.keras.layers.Layer):
    def __init__(self, dim=8, trainable=False):
        self.dim = dim
        self.trainable = trainable
        init_weight = np.reshape(np.identity(self.dim), (self.dim, 1, self.dim))
        self.initializer = tf.keras.initializers.Constant(init_weight)
        super(time_delayed_embed, self).__init__(trainable=self.trainable)

    def build(self, input_shape):
        self.conv1 = Conv1D(
            filters=self.dim,
            kernel_size=self.dim,
            padding="causal",
            use_bias=False,
            trainable=self.trainable,
            kernel_initializer=self.initializer,
        )

    @tf.function
    def call(self, input):
        x = self.conv1(input)
        return x


class vec_layer(tf.keras.layers.Layer):
    def __init__(self, outdim=8):
        super(vec_layer, self).__init__(trainable=True)
        self.outdim = outdim

    def build(self, input_shape):
        self.wa = self.add_weight(
            name="wa",
            shape=(1, self.outdim),
            initializer="uniform",
            trainable=True,
        )

    def call(self, x):
        return K.dot(x, self.wa)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class TSTransformer(tf.keras.Model):
    def __init__(
        self,
        t2v_size,
        embed_size,
        window_size,
        num_layers,
        d_model,
        num_heads,
        dff,
        rate=0.1,
        flex_embed=False,
    ):
        super(TSTransformer, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = time_delayed_embed(embed_size, trainable=flex_embed)
        self.t2vec = Time2Vec(seq_length=window_size, kernel_size=t2v_size-1)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_out = tf.keras.layers.Dense(1)
        self.mask = create_look_ahead_mask(window_size)

    @tf.function
    def call(self, x, training):

        seq_len = tf.shape(x)[1]
        # input x shape: bs x wnidow x 1
        x1 = self.embedding(x)  # bs x window x embed_size
        x2 = self.t2vec(x) # 1 X window x t2vec_size

        # cheat a bit by using broadcasting
        x = tf.keras.layers.Concatenate(axis=-1)([x1, x2 + x1[:,:,0:1] * 0.])
        # x = x1
        # bs x window x (embed_size + t2vec_size)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training, self.mask)

        # x.shape == (batch_size, window, d_model)
        x = self.final_out(x)
        return x

if __name__ == "__main__":
    xin = np.arange(32).reshape((1,-1,1)).astype(float)
    l = time_delayed_embed(7)
    t = Conv1D(kernel_size=8, filters=4, use_bias=False)
    b = t(xin)
    xout = l(xin)
    print(xout)