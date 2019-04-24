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


global_name = "20190419_test.json"

block_methods=[
    "basic_block",
    "double_basic"
]

double_input=[
    True,False
]
concatenate=[
    "half_concanate",
    "full_concanate",
    "sum"
]

relu_option = False
epochs = 10
split = 1.0
batch_size = 32
dataset = mnist_dataset(is_zero_center=True)

for m in block_methods:
    for input_mode in double_input:
        if m == "double_basic":
            for concate_mode in concatenate:
                option = {
                    "relu_option": relu_option,
                    "double_input": input_mode,
                    "concatenate": concate_mode,
                    "block": m
                }

                model, tester = make(option,dataset,batch_size,epochs,split)
                run(model,tester,global_name)
        else:
            option = {
                    "relu_option": relu_option,
                    "double_input": input_mode,
                    "block": m,
                    "concatenate": "none"
            }
            model, tester = make(option,dataset,batch_size,epochs,split)
            run(model,tester,global_name)




