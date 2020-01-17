from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as datasets

# Load data

# 60% / 40%
train_validation_split = datasets.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = datasets.load(
    name="imdb_reviews",
    split=(train_validation_split, datasets.Split.TEST),
    as_supervised=True
)

(train_examples_batch, train_labels_batch) = next(iter(train_data.batch(10)))

# Embeddings
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# Build the full model
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_data.shuffle(10000).batch(512), epochs=20, validation_data=validation_data.batch(512), verbose=1)

# Evaluate model
results = model.evaluate(test_data.batch(512), verbose=2)