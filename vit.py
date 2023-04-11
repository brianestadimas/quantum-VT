import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

# Data Preparation and seeding
num_classes = 100
input_shape = (32, 32, 3)

# Load CIFAR-100
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()


# Data preprocessing & Transformer class
class TransformerBlock():
    def __init__(self):
        self.image_size = 72  # will resize input images to this size
        self.patch_size = 6  # size of the patches to be extract from the input images
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]  # Size of the transformer layers
        self.transformer_layers = 8
        self.mlp_head_units = [2048, 1024]  # dense layer for final classifier
        
    def data_augmentation(self):
        data_augmentation = keras.Sequential(
            [
                layers.Normalization(),
                layers.Resizing(self.image_size, self.image_size),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=0.02),
                layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            ],
            name="data_augmentation",
        )
        # Compute the mean and the variance of the training data for normalization.
        data_augmentation.layers[0].adapt(x_train)
        return data_augmentation

    # Optional
    def data_augmentation_2(self):
        data_augmentation = keras.Sequential(
            [
                layers.Resizing(self.image_size, self.image_size),
                layers.RandomFlip("vertical"),
                layers.RandomRotation(factor=0.12),
                layers.Resizing(self.image_size, self.image_size),
                layers.RandomZoom(height_factor=0.35, width_factor=0.35),
                layers.RandomContrast(factor=0.2),
                layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
            ],
            name="data_augmentation",
        )
        # Compute the mean and the variance of the training data for normalization.
        data_augmentation.layers[0].adapt(x_train)
        return data_augmentation
        
    def get_sample_images(self):
        plt.figure(figsize=(4, 4))
        image = x_train[np.random.choice(range(x_train.shape[0]))]
        plt.imshow(image.astype("uint8"))
        plt.axis("off")

        resized_image = tf.image.resize(
            tf.convert_to_tensor([image]), size=(self.image_size, self.image_size)
        )
        patches = Patches(self.patch_size)(resized_image)
        print(f"Image size: {self.image_size} X {self.image_size}")
        print(f"Patch size: {self.patch_size} X {self.patch_size}")
        print(f"Patches per image: {patches.shape[1]}")
        print(f"Elements per patch: {patches.shape[-1]}")

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[0]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (self.patch_size, self.patch_size, 3))
            plt.imshow(patch_img.numpy().astype("uint8"))
            plt.axis("off")

    # Begin the transformer model
    def create_vit_classifier(self):
        inputs = layers.Input(shape=input_shape)
        # Augment data.
        augmented = self.data_augmentation()
        augmented = augmented(inputs)
        
        # Create patches.
        patches = Patches(self.patch_size)(augmented)
    
        # Encode patches.
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = MLP.construct_layer(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = MLP.construct_layer(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model
    

## Implement the patch encoding layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


## Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# Implement the MLP Layer
class MLP():
    def construct_layer(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x


if __name__ == '__main__':
    # Configure the hyperparameters
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 256
    num_epochs = 4
    
    transformer = TransformerBlock()
    
    # Print sample images
    # transformer.get_sample_images()
    
    # Create the model
    model = transformer.create_vit_classifier()
    
    # Compile model
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    