from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as datasets

import numpy as np

(train_data, test_data), info = datasets.load(
    name="imdb_reviews/subwords8k",
    split=(datasets.Split.TRAIN, datasets.Split.TEST),
    as_supervised=True,
    with_info=True
)

encoder = info.features['text'].encoder

# Exploration of the data
# for train_example, train_label in train_data.take(1):
#     print("Encoded text:", train_example[:10].numpy())
#     print("Label:", train_label.numpy())

# Data sanitization
BUFFER_SIZE = 1000
# The docs are broken on google.
# Found the solution on github to use the compat.v1.data.get_output_shapes method
train_output_shapes = tf.compat.v1.data.get_output_shapes(train_data)
train_batches = (train_data.shuffle(BUFFER_SIZE).padded_batch(32, train_output_shapes))
test_batches = (test_data.shuffle(BUFFER_SIZE).padded_batch(32, train_output_shapes))

# for example_batch, label_batch in train_batches.take(2):
#     print("Batch shape:", example_batch.shape)
#     print("Label shape:", label_batch.shape)
#

model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(train_batches, epochs=10, validation_data=test_batches, validation_steps=30)

loss, accuracy = model.evaluate(test_batches)

print(loss, accuracy)