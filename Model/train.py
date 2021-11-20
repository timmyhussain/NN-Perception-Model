# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:53:33 2021

@author: user
"""
import tensorflow as tf
import tensorflow.nn as nn
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pptk
import seaborn as sns
import matplotlib.pyplot as plt
# import pydot

split = [0.7, 0.2, 0.1]

data_path = "training_data_1/"
map_path = "map1/"

all_examples = np.load(data_path+"Lidar_Training_Data.npy")
# all_examples = all_examples.reshape((-1, 720))
all_labels = np.load(data_path+"Position_Training_Data.npy")
all_labels = all_labels.reshape((all_labels.shape[0], -1))
map_data = np.load(map_path+"grid_map.npy")
map_data = map_data.flatten()
N = all_examples.shape[0]
N_train = int(split[0]*N)

train_examples = all_examples[:int(N_train), :, :]
train_labels = all_labels[:int(N_train), :]

test_examples = all_examples[int(N_train):, :]
test_labels = all_labels[int(N_train):, :]

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))


BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


lidar_input = keras.Input(shape=(720,1), name="lidar")
map_input = keras.Input(shape=(100,), name="map")

lidar_features = layers.Conv1D(360, 2, input_shape=(None,720,1))(lidar_input)
lidar_features = layers.Dense(256, activation="relu")(lidar_features)
lidar_features = layers.Dense(128, activation = "relu")(lidar_features)

x = layers.concatenate([lidar_features, map_input])

x = layers.Dense(100, name="belief", activation="softmax")(x)

model = keras.Model(
        inputs = [lidar_input, map_input],
        outputs=[x])


keras.utils.plot_model(model, "model.png", show_shapes=(True))

# model = keras.Sequential([keras.layers.Flatten(input_shape=(720,1)),
#                          keras.layers.Dense(256, activation="relu"),
#                          keras.layers.Dense(100, activation="relu")])

optimizer = keras.optimizers.Adam()
loss = {"belief": keras.losses.CategoricalCrossentropy(from_logits=False)}
metrics = ["sparse_categorical_accuracy"]
model.compile(optimizer=optimizer, loss=loss)#, metrics=metrics)
model.fit(
    {"lidar_input": train_examples, "map_input": np.tile(map_data, N_train).reshape((-1, 100))} ,
    {"belief": train_labels},
    epochs=100,
    batch_size=32)
# # arr = np.load("Data3/Lidar_Data.npy", allow_pickle=True)

def random_check(i):
    # arr = all_examples[i].reshape()
    # pptk.viewer(np.hstack((arr[i], np.zeros((arr[i].shape[0], 1)))))
    fig, ax = plt.subplots(1, 2)
    sns.heatmap(model.predict({"lidar_input": test_examples[i].reshape(1, -1), "map_input": map_data.reshape(1, -1)}).reshape((10,10)), ax=ax[0])
    ax[0].set_title("Predicted Belief")
    sns.heatmap(test_labels[i].reshape((10,10)), ax=ax[1])
    ax[1].set_title("Actual Position")
    
    
# print(len(arr))
# random_check(5)