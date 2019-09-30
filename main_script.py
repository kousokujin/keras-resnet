import numpy as np
import sklearn.metrics as metrics
import json
import os
from model_module import ModelBuilder
from ResNetTester_cls import ResNetTester,make,check_run
from Datasets import (cifer10_datasets,cifer100_datasets,mnist_dataset)


global_name = "20190924"

relu_option = False
epochs = 1
split = 1.0
batch_size = 32
dataset = cifer10_datasets(is_zero_center=True)
json_file = "20190924_options_bk.json"

json_experience = ""
with open(json_file) as f:
    s = f.read()
    json_experience = json.loads(s)

for e in json_experience["options"]:
    isRuned = check_run(e,'result/'+global_name+'.json')

    if isRuned == False:
        model, tester = make(e,dataset,batch_size,epochs,split)
        tester.run(model,global_name)
        del tester
    else:
        print("skiped")
