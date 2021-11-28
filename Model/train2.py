# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 23:19:07 2021

@author: user
"""
import json
from helper import Net

map_size = 50
data_path = "Final_{map_size}{map_size}/".format(map_size=map_size)
map_path = "../Maps/map1-new/"

paths = [data_path, map_path]
activations = ["tanh", "tanh", "tanh"]
net = Net(paths, "NN4", map_size, activations)
history = net.train(10000, 64)
with open(net.path+net.name+'_history.json', 'w') as fp:
    json.dump(history.history, fp)