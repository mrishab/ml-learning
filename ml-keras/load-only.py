from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import os

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


# Loading
checkpoint_dir = os.path.abspath("models/save-and-load.ckpt")

# Training
model = build_model()

# Evaluation untrained model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Loss:", loss, "Accuracy:", acc * 100)

# Reevaluation after loading a trained model
model.load_weights(checkpoint_dir)
loss_trained, accuracy_trained = model.evaluate(test_images, test_labels, verbose=2)
print("Loss:", loss_trained, "Accuracy:", accuracy_trained * 100)

print ("Accuracy improved by:", accuracy_trained - acc)