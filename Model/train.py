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

split = [1, 0.2, 0.1]

data_path = "../Data/training_data_1/"
data_path = "Final_1010/"
map_path = "../Maps/map1-new/"

all_examples = np.load(data_path+"Lidar_data.npy")
# all_examples = all_examples.reshape((-1, 30, 24))
all_labels = np.load(data_path+"Pose_data.npy")
all_labels = all_labels.reshape((all_labels.shape[0], -1))
orientation_labels = np.load(data_path+"Orientation_data.npy")
orientation_labels = orientation_labels.reshape((orientation_labels.shape[0], -1))



#binary/determinsitic label 1s and 0s
# noise = 0
# noise = np.random.uniform(0, 0.05)
noise = 1e-3


map_data = np.load(map_path+"map_1010.npy")

#uniform belief map input
# map_data = map_data - 1
# map_data = map_data/np.sum(map_data)


# map_data = map_data.flatten()

all_labels = all_labels + noise - noise*map_data.flatten()
all_labels = all_labels/all_labels.sum(axis=1, keepdims=True)

# all_labels = all_labels+1
N = all_examples.shape[0]
N_train = int(split[0]*N)

train_examples = all_examples[:int(N_train), :, :]
train_labels = all_labels[:int(N_train), :]

test_examples = all_examples[int(N_train):, :]
test_labels = all_labels[int(N_train):, :]

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))


# BATCH_SIZE = 64
# SHUFFLE_BUFFER_SIZE = 100
# train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


lidar_input = keras.Input(shape=(720,), name="lidar")
map_input = keras.Input(shape=(12, 12,), name="map")

# lidar_features = layers.Conv1D(24, 2, input_shape=(None, 30, 24))(lidar_input)
# lidar_features = layers.MaxPool1D(2)(lidar_features)
# lidar_features = layers.Conv1D(24, 2, input_shape=(None, 29, 24))(lidar_features)
# lidar_features = layers.MaxPool1D(2)(lidar_features)
# lidar_features = layers.Flatten()(lidar_features)

# lidar_features = layers.Flatten()(lidar_input)

#did well with these off
#did well with these on and tanh
#did as good/better with these on and relu
# lidar_features = layers.Dense(720, activation="relu")(lidar_input)
lidar_features = layers.Dense(360, activation="relu")(lidar_input)
lidar_features = layers.Dense(180, activation="relu")(lidar_features)
# print(lidar_features.shape)

#did well with relu
lidar_features = layers.Dense(144, activation="relu")(lidar_features)
lidar_features = layers.Dense(72, activation = "relu")(lidar_features)

map_features = layers.Conv1D(12, 2, input_shape=(None, 12, 12))(map_input)
map_features = layers.Conv1D(12, 2, input_shape=(None, 12, 12))(map_features)
map_features = layers.MaxPool1D(2)(map_features)
map_features = layers.Flatten()(map_features)
# print(map_features.shape)
x = layers.concatenate([lidar_features, map_features])
x = layers.Dense(132, activation = "relu")(x)

#does well with dropout
#does pretty well without it
# x = layers.Dropout(0.1)(x)
position = layers.Dense(144, name="position_belief", activation="softmax")(x)
orientation = layers.Dense(144, name="orientation_belief")(x)

model = keras.Model(
        inputs = [lidar_input, map_input],
        outputs=[position, orientation])

keras.utils.plot_model(model, "model.png", show_shapes=(True))

# model = keras.Sequential([keras.layers.Flatten(input_shape=(720,1)),
#                          keras.layers.Dense(256, activation="relu"),
#                          keras.layers.Dense(100, activation="relu")])

optimizer = keras.optimizers.Adam()

#different loss function
# loss = {"belief": keras.losses.CategoricalCrossentropy(from_logits=False)}
loss = {"position_belief": keras.losses.MeanSquaredError(),
        "orientation_belief": keras.losses.MeanSquaredError()}#did ok with mse

metrics = ["sparse_categorical_accuracy"]
model.compile(optimizer=optimizer, loss=loss)#, metrics=metrics)
history = model.fit(
    {"lidar_input": train_examples, "map_input": np.tile(map_data, N_train).reshape((-1, 12, 12))} ,
    {"position_belief": train_labels, "orientation_belief": orientation_labels},
    epochs=100,
    batch_size=64)
# # arr = np.load("Data3/Lidar_Data.npy", allow_pickle=True)

def random_check(i):
    # map_data = np.load(map_path+"grid_map.npy")
    map_data = np.load(map_path+"map_1010.npy")

    # arr = all_examples[i].reshape()
    # pptk.viewer(np.hstack((arr[i], np.zeros((arr[i].shape[0], 1)))))
    # plt.figure(figsize=[10,5])
    fig, ax = plt.subplots(1, 3, figsize=[15,5])
    sns.heatmap(model.predict({"lidar_input": train_examples[i].reshape(1, 720), "map_input": map_data.reshape(1, 12, 12)})[0].reshape((12,12))+-1*map_data, ax=ax[0])
    ax[0].set_title("Position Belief")
    sns.heatmap(train_labels[i].reshape((12,12))+-1*map_data, ax=ax[1])
    ax[1].set_title("Actual Position")
    sns.heatmap(model.predict({"lidar_input": train_examples[i].reshape(1, 720), "map_input": map_data.reshape(1, 12, 12)})[1].reshape((12,12)), ax=ax[2])
    ax[2].set_title("Orientation Belief")
    
    
# print(len(arr))
# random_check(5)