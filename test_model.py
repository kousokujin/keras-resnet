import numpy as np
import sklearn.metrics as metrics
import json
import os
from ModelBuilder import ResnetBuilder
from plot_result import (plot_acc_history,plot_loss_history,outputfile_evalute,get_time,output_text)
from ResNetTester_cls import ResNetTester
from Datasets import (cifer10_datasets,cifer100_datasets,mnist_dataset)
from json_writer import json_write
from converter_json import (json_bar2_graph,include_dict)
from graph_plot.plot import plot_drawer
from graph_plot.json_graph import json_plot_data


def make(option,dataset,batch_size,epochs,split):
    tester = ResNetTester(option = option)
    tester.setDataset(dataset)
    tester.batch_size = batch_size
    tester.nb_epoch = epochs
    tester.validation_split = split
    model = ResnetBuilder.build_manual(input_shape=dataset.get_shape(),num_outputs=dataset.get_categorical(), option = option)
    return model,tester

def run(model,tester,global_name):
    tester.run_model(model)
    tester.evalute_model()
    json_write(tester,'result/'+global_name)

relu_option = False
epochs = 10
split = 1.0
batch_size = 32
dataset = cifer10_datasets(is_zero_center=True)
global_name = "20190620_1_test.json"

option = {
    "relu_option": False,
    "double_input": False,
    "block": "basic_block",
    "concatenate": "none",
    "reseption" :[2,2,2,2]
}

model, tester = make(option,dataset,batch_size,epochs,split)
run(model,tester,global_name)