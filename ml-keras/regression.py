from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import seaborn as sns

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

import matplotlib.pyplot as plt
import pathlib

import numpy as np
import pandas as pd

dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
dataset = dataset.dropna()

# One hot encoding for the origin
dataset['Origin'] = dataset['Origin'].map(lambda x: {1: "USA", 2: "Europe", 3: "Japan"}.get(x))
dataset = pd.get_dummies(dataset, prefix="", prefix_sep="")

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Exploration
plot = sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
# print(train_stats)

# Split features and labels
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")


# Normalize the integer values
def norm(ds):
    return (ds - train_stats['mean']) / train_stats['std']


norm_train_dataset = norm(train_dataset)
norm_test_dataset = norm(test_dataset)


# Build the model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=[len(train_dataset.keys())]),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1)
    ])

    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])

    return model


model = build_model()

# Training
# # Training without early stopping
# history = model.fit(norm_train_dataset, train_labels, epochs=1000,
#                     validation_split=0.2, verbose=0,
#                     callbacks=[tfdocs.modeling.EpochDots()])
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail())

# Adding automatic callback to stop training
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
early_history = model.fit(norm_train_dataset, train_labels, epochs=1000,
                          validation_split=0.2, verbose=0,
                          callbacks=[early_stop, tfdocs.modeling.EpochDots()])

# Testing
loss, mae, mse = model.evaluate(norm_test_dataset, test_labels, verbose=2)

print(loss, mae, mse)