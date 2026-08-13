"""Adaptive Attention U-Net (Hybrid CNN-Transformer)

This module implements a U-Net backbone with lightweight Transformer-style
attention blocks in the decoder. The attention blocks use Keras MultiHeadAttention
with LayerNormalization, a small feed-forward network and an adaptive gating
mechanism that lets the network learn how much to rely on global attention
vs local convolutional features.

The module exposes:
- build_adaptive_attention_unet(input_shape=(256,256,1), num_heads=4, embed_dim=128)
- dice_coefficient, iou_score, combined_loss

Includes a small demo when executed as __main__ to verify model builds.
"""
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def conv_block(x, filters, kernel_size=3, activation='relu'):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def down_block(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(c)
    return c, p


def up_concat(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip])
    return x


def transformer_block_2d(x, num_heads=4, embed_dim=128, mlp_dim=256, dropout=0.0):
    """A lightweight transformer-like block for 2D feature maps.

    - Projects features to embed_dim via 1x1 conv
    - Flattens spatial dims to sequence and applies MultiHeadAttention
    - Adds residuals, LayerNorm and a small MLP (implemented as Conv1x1 -> Depthwise/Conv -> Conv1x1)
    - Reshapes back to spatial map
    """
    # x: (B, H, W, C)
    shape = tf.shape(x)
    B, H, W = shape[0], shape[1], shape[2]

    # inject a small coordinate map as positional hint (normalized coordinates)
    coords_y = tf.linspace(-1.0, 1.0, H)
    coords_x = tf.linspace(-1.0, 1.0, W)
    yy = tf.reshape(coords_y, (H, 1)) * tf.ones((1, W), dtype=tf.float32)
    xx = tf.reshape(coords_x, (1, W)) * tf.ones((H, 1), dtype=tf.float32)
    yy = tf.expand_dims(tf.expand_dims(yy, 0), -1)  # (1,H,W,1)
    xx = tf.expand_dims(tf.expand_dims(xx, 0), -1)  # (1,H,W,1)
    yy = tf.tile(yy, [B, 1, 1, 1])
    xx = tf.tile(xx, [B, 1, 1, 1])

    x_with_coords = tf.concat([x, xx, yy], axis=-1)

    # Project to embedding
    proj = layers.Conv2D(embed_dim, 1, padding='same')(x_with_coords)

    # Flatten spatial dims to sequence
    seq = layers.Reshape((H * W, embed_dim))(proj)

    # Multi-head self-attention (query=key=value=seq)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)(seq, seq)

    # Residual + Norm
    seq = layers.Add()([seq, attn_output])
    seq = layers.LayerNormalization(epsilon=1e-6)(seq)

    # Feed-forward network (MLP)
    ff = layers.Dense(mlp_dim, activation='gelu')(seq)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(embed_dim)(ff)
    seq = layers.Add()([seq, ff])
    seq = layers.LayerNormalization(epsilon=1e-6)(seq)

    # Reshape back to spatial map
    out = layers.Reshape((H, W, embed_dim))(seq)
    return out


def adaptive_attention_block(x, num_heads=4, embed_dim=128, mlp_dim=256):
    """Apply transformer_block_2d followed by adaptive gating and residual addition.

    The gate learns how much of the attended features to let through.
    """
    # Save input for residual connection
    input_channels = tf.shape(x)[-1]

    # Transformer style processing
    attn = transformer_block_2d(x, num_heads=num_heads, embed_dim=embed_dim, mlp_dim=mlp_dim)

    # Project attention output to match input channels if needed
    attn_projected = layers.Conv2D(input_channels, 1, padding='same')(attn)

    # Gate computed from concatenation of original features and attention
    gate_input = layers.Concatenate()([x, attn_projected])
    gate = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(gate_input)

    gated = layers.Multiply()([attn_projected, gate])

    out = layers.Add()([x, gated])
    out = layers.LayerNormalization(epsilon=1e-6)(out)
    return out


def build_adaptive_attention_unet(input_shape: Tuple[int, int, int] = (256, 256, 1),
                                  base_filters: int = 32,
                                  num_heads: int = 4,
                                  embed_dim: int = 128,
                                  mlp_dim: int = 256) -> Model:
    inputs = Input(shape=input_shape)

    # Encoder
    c1, p1 = down_block(inputs, base_filters)
    c2, p2 = down_block(p1, base_filters * 2)
    c3, p3 = down_block(p2, base_filters * 4)
    c4, p4 = down_block(p3, base_filters * 8)

    # Bridge
    b = conv_block(p4, base_filters * 16)

    # Decoder with adaptive attention inserted before conv blocks
    u6 = up_concat(b, c4, base_filters * 8)
    u6 = adaptive_attention_block(u6, num_heads=num_heads, embed_dim=embed_dim, mlp_dim=mlp_dim)
    c6 = conv_block(u6, base_filters * 8)

    u7 = up_concat(c6, c3, base_filters * 4)
    u7 = adaptive_attention_block(u7, num_heads=num_heads, embed_dim=embed_dim, mlp_dim=mlp_dim)
    c7 = conv_block(u7, base_filters * 4)

    u8 = up_concat(c7, c2, base_filters * 2)
    u8 = adaptive_attention_block(u8, num_heads=num_heads, embed_dim=embed_dim, mlp_dim=mlp_dim)
    c8 = conv_block(u8, base_filters * 2)

    u9 = up_concat(c8, c1, base_filters)
    u9 = adaptive_attention_block(u9, num_heads=num_heads, embed_dim=embed_dim, mlp_dim=mlp_dim)
    c9 = conv_block(u9, base_filters)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs, outputs, name='AdaptiveAttentionUNet')
    return model


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = 1 - dice_coefficient(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice


if __name__ == '__main__':
    # Quick sanity check: build model and run a forward pass with dummy data
    model = build_adaptive_attention_unet(input_shape=(128, 128, 1), base_filters=16, num_heads=4, embed_dim=64, mlp_dim=128)
    model.summary()

    # Dummy forward pass
    x = np.random.rand(2, 128, 128, 1).astype(np.float32)
    y = model.predict(x)
    print('Output shape:', y.shape)
