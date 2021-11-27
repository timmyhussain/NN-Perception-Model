# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 12:52:26 2021

@author: user
"""
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class Net:
    def __init__(self, paths, name, map_size, activations, noise=1e-3, load_model=False):
        self.name = name
        self.path = name+"/"
        self.get_data(paths, map_size, noise)
        if not os.path.exists(self.path):
             os.makedirs(self.path)  
        self.checkpoint_dir = self.path+"checkpoints/" 
        self.checkpoint_path = self.path+"checkpoints/cp.ckpt"
        self.create_model(activations)
        if os.path.exists(self.checkpoint_dir):
            self.model.load_weights(self.checkpoint_path)
        pass
    
    def get_data(self, paths, map_size, noise):
        data_path, map_path = paths
        self.train_examples = np.load(data_path+"Lidar_data.npy")
        position_labels = np.load(data_path+"Pose_data.npy")
        self.train_position_labels = position_labels.reshape((position_labels.shape[0], -1))
        orientation_labels = np.load(data_path+"Orientation_data.npy")
        self.train_orientation_labels = orientation_labels.reshape((orientation_labels.shape[0], -1))
        
        N = self.train_position_labels.shape[0]
        self.map = np.load(map_path+"map_{map_size}{map_size}.npy".format(map_size=map_size))
        
        #random ordering each time network is initialized
        shuffler = np.random.permutation(N)
        self.train_examples = self.train_examples[shuffler]
        self.train_position_labels = self.train_position_labels[shuffler]
        self.train_orientation_labels = self.train_orientation_labels[shuffler]
        
        #add noise to final belief distribution so no pure 0s
        self.train_position_labels = self.train_position_labels + noise - noise*self.map.flatten()
        self.train_position_labels = self.train_position_labels/self.train_position_labels.sum(axis=1, keepdims=True)
        self.map_examples = np.tile(self.map, N).reshape((-1, *self.map.shape))
        pass
    
    def create_model(self, activations):
        act1, act2, act3 = activations
        
        #inputs
        lidar_input = keras.Input(shape=(720,), name="lidar")
        map_input = keras.Input(shape=(*self.map.shape,), name="map")
        
        lidar_features = layers.Dense(360, activation=act1)(lidar_input)
        lidar_features = layers.Dense(180, activation=act1)(lidar_features)
        
        lidar_features = layers.Dense(144, activation=act2)(lidar_features)
        lidar_features = layers.Dense(72, activation = act2)(lidar_features)
        
        #feature extraction for map
        map_features = layers.Conv1D(self.map.shape[0], 2, input_shape=(None, *self.map.shape))(map_input)
        map_features = layers.Conv1D(self.map.shape[0], 2, input_shape=(None, *self.map.shape))(map_features)
        map_features = layers.MaxPool1D(2)(map_features)
        map_features = layers.Flatten()(map_features)
        
        x = layers.concatenate([lidar_features, map_features])
        x = layers.Dense(132, activation = act3)(x)
        x = layers.Dropout(0.1)(x)
        
        #outputs
        position = layers.Dense(self.map.shape[0]*self.map.shape[1], name="position_belief", activation="softmax")(x)
        orientation = layers.Dense(self.map.shape[0]*self.map.shape[1], name="orientation_belief")(x)
        
        self.model = keras.Model(
                inputs = [lidar_input, map_input],
                outputs=[position, orientation])
        
        optimizer = keras.optimizers.Adam()

        loss = {"position_belief": keras.losses.MeanSquaredError(),
                "orientation_belief": keras.losses.MeanSquaredError()}#did ok with mse
        
        
        self.model.compile(optimizer=optimizer, loss=loss)#, metrics=metrics)
        
        keras.utils.plot_model(self.model, self.path+self.name+"_model.png", show_shapes=(True))

        pass
    
    def train(self, n_epochs, batch_size):
               
        cp_callback = keras.callbacks.ModelCheckpoint(self.checkpoint_path, save_weights_only=True)

        history = self.model.fit(
            {"lidar_input": self.train_examples, "map_input": self.map_examples} ,
            {"position_belief": self.train_position_labels, "orientation_belief": self.train_orientation_labels},
            epochs=n_epochs,
            batch_size=batch_size,
            callbacks=[cp_callback])
        return history
    
    def evaluate(self, inputs):
        lidar_input, map_input = inputs
        return self.model.predict({"lidar_input": lidar_input.reshape(1, 720), "map_input": map_input.reshape(1, *self.map.shape)})