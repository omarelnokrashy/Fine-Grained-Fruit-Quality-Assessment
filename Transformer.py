import tensorflow as tf
from tensorflow.keras import layers
import math
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim #added embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training=False): 
        attn_output = self.att(inputs, inputs, training=training) 
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class VisionTransformer(tf.keras.Model):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, num_heads, ff_dim, num_layers, rate=0.1, channels=3): # Added channels
        super(VisionTransformer, self).__init__()
        self.image_size = image_size 
        self.patch_size = patch_size 
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = (patch_size * patch_size * channels) 
        self.embed_dim = embed_dim
        self.num_classes = num_classes 
        self.channels = channels
        self.num_layers = num_layers 


        # Patch embedding layer
        self.patch_embed = layers.Dense(embed_dim)

        # Positional embedding
        self.pos_embed = self.add_weight(
            shape=(1, self.num_patches, embed_dim),
            initializer="random_normal",
            trainable=True,
        )

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)
        ]

        # Classification head
        self.mlp_head = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(128, activation="relu"),
            layers.Dropout(rate),
            layers.Dense(num_classes, activation="softmax")
        ])



    def build(self, input_shape): 
        super(VisionTransformer, self).build(input_shape)
        if not self.mlp_head.built:
            self.mlp_head.build(input_shape=(None, self.embed_dim)) 

    def extract_patches(self, images, patch_size):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, inputs, training=False): 
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=1)
        patches = self.extract_patches(inputs, patch_size=self.patch_size)

        # Embed patches
        x = self.patch_embed(patches)

        # Add positional embedding
        x += self.pos_embed

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training) 

        # Global average pooling
        x = tf.reduce_mean(x, axis=1)

        # Classification head
        return self.mlp_head(x)

    def get_config(self):
        config = super(VisionTransformer, self).get_config()
        config.update({
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'num_heads': 8,
            'ff_dim': 256,
            'num_layers': self.num_layers,
            'rate': 0.4,
            'channels': 3,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def Create_Transformer(image_size, num_classes, channels=3):
    """
    Creates a hybrid CNN-ViT model.  The CNN acts as a feature extractor,
    and the ViT processes the extracted features.

    Args:
        image_size: The size of the input image.
        num_classes: The number of classes for classification.
        channels: Number of channels
    Returns:
        A tf.keras.Model instance representing the hybrid model.
    """
    # CNN Feature Extractor
    cnn_model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, channels)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)), 
    ])

    # Determine the shape of the CNN output
    cnn_output_shape = cnn_model.output_shape
    cnn_output_size = cnn_output_shape[1]
    cnn_output_channels = cnn_output_shape[3]

    
    # ViT Parameters
    patch_size = 1  
    embed_dim = 256
    num_heads = 8
    ff_dim = 256
    num_layers = 6
    dropout_rate = 0.4

    # ViT Model
    vit_model = VisionTransformer(
        image_size=cnn_output_size,  
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        rate=dropout_rate,
        channels=cnn_output_channels
    )

    # Hybrid Model
    inputs = tf.keras.Input(shape=(image_size, image_size, channels))
    cnn_features = cnn_model(inputs)
    # Reshape the CNN output to be compatible with ViT's patch embedding
    x = layers.Reshape(target_shape=(cnn_output_size * cnn_output_size, cnn_output_channels))(cnn_features)
    outputs = vit_model(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

