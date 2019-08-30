import numpy as np
import sklearn.metrics as metrics
import json
import os
from model_module import ModelBuilder
from ResNetTester_cls import ResNetTester,make,run
from Datasets import (cifer10_datasets,cifer100_datasets,mnist_dataset)
from tools import json_writer
from graph_plot.plot import plot_drawer
from graph_plot.json_graph import json_plot_data


relu_option = False
epochs = 1
split = 0.1
batch_size = 128
dataset = mnist_dataset(is_zero_center=True)
global_name = "test"

option = {
    "relu_option": False,
    "double_input": False,
    "block": "basic_block",
    "concatenate": "none",
    "reseption" :[2,2,2,2],
    "dropout": 0
}

model, tester = make(option,dataset,batch_size,epochs,split)
run(model,tester,global_name)