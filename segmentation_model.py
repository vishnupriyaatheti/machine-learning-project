import tensorflow as tf
from tensorflow.keras import layers


def conv_block(x, filters, kernel_size=3, activation="relu"):
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        # x: (batch, seq_len, embed_dim)
        attn_out = self.attn(x, x)
        x = x + attn_out
        x = self.ln1(x)
        mlp_out = self.mlp(x, training=training)
        x = x + mlp_out
        x = self.ln2(x)
        return x


def patchify(x, patch_size):
    # x: (batch, H, W, C)
    # produce patches via Conv2D with stride=patch_size
    return layers.Conv2D(filters=x.shape[-1], kernel_size=patch_size, strides=patch_size)(x)


def build_hybrid_unet(
    input_shape=(256, 256, 3),
    num_classes=1,
    base_filters=32,
    depth=4,
    transformer_depth=4,
    num_heads=8,
    mlp_dim=512,
    patch_size=2,
):
    """Builds a U-Net with a Transformer bottleneck.

    Encoder: repeated Conv blocks + MaxPool
    Bottleneck: flatten patches -> Transformer blocks -> reshape to spatial map
    Decoder: upsampling, skip connections
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    skips = []
    x = inputs
    for i in range(depth):
        filters = base_filters * (2 ** i)
        x = conv_block(x, filters)
        skips.append(x)
        x = layers.MaxPool2D(pool_size=2)(x)

    # Bottleneck conv to reduce channels if needed
    filters = base_filters * (2 ** depth)
    x = conv_block(x, filters)

    # Prepare patches for transformer
    # Ensure H and W are divisible by patch_size
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[3]
    if h is None or w is None:
        # dynamic shape fallback: compute using tf.shape when calling; here we assume static known
        pass

    # Patch embedding using a Conv2D with stride = patch_size
    patch_embed = layers.Conv2D(filters=c, kernel_size=patch_size, strides=patch_size, padding="valid")(x)
    # flatten patches
    seq_len = (patch_embed.shape[1] or 1) * (patch_embed.shape[2] or 1)
    embed_dim = patch_embed.shape[-1]
    flat = layers.Reshape((seq_len, embed_dim))(patch_embed)

    # Add learnable positional embeddings
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos = pos_emb(positions)
    flat = flat + pos

    # Transformer blocks
    for _ in range(transformer_depth):
        flat = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)(flat)

    # Project back to spatial map
    x = layers.Reshape((patch_embed.shape[1], patch_embed.shape[2], embed_dim))(flat)
    # If patch_size>1, upsample back to bottleneck resolution
    if patch_size > 1:
        x = layers.UpSampling2D(size=patch_size, interpolation="nearest")(x)
        # adjust channels if mismatch
        if x.shape[-1] != filters:
            x = layers.Conv2D(filters, 1, padding="same")(x)

    # Decoder
    for i in reversed(range(depth)):
        skip = skips[i]
        filters = base_filters * (2 ** i)
        x = layers.UpSampling2D(size=2, interpolation="nearest")(x)
        # If spatial sizes mismatch due to rounding, crop or pad (here we rely on matching shapes)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)

    # Output
    if num_classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"

    outputs = layers.Conv2D(num_classes, kernel_size=1, padding="same", activation=activation)(x)

    model = tf.keras.Model(inputs, outputs, name="hybrid_unet_vit")
    return model


if __name__ == "__main__":
    m = build_hybrid_unet()
    m.summary()
