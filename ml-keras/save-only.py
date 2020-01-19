from __future__ import absolute_import, division, print_function, unicode_literals
import os

from tensorflow import keras

# Data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation="relu", input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    return model


# Saving callbacks
checkpoint_dir = os.path.abspath("models/save-and-load.ckpt")
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, verbose=1)

# Training
model = build_model()

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[checkpoint_cb])
