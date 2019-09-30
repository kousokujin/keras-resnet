import numpy as np
import sklearn.metrics as metrics
import json
import os
import datetime
from model_module import ModelBuilder
from ResNetTester_cls import ResNetTester,make,check_run
from Datasets import (cifer10_datasets,cifer100_datasets,mnist_dataset)
from tools import converter_json


global_name = "20190930_test"
loop = 3

relu_option = False
epochs = 1
split = 1.0
batch_size = 32
dataset = cifer10_datasets(is_zero_center=True)
json_file = "options/20190930_options_bk.json"

result_file = []

for i in range(loop):
    global_loop_name = global_name + str(i)
    json_experience = ""
    with open(json_file) as f:
        s = f.read()
        json_experience = json.loads(s)

    for e in json_experience["options"]:
        filename = 'result/'+global_loop_name+'.json'
        isRuned = check_run(e,filename)
        result_file.append(filename)

        if isRuned == False:
            model, tester = make(e,dataset,batch_size,epochs,split)
            tester.run(model,global_loop_name)
            del tester
        else:
            print("skiped")

all_result = []
for e in json_experience["options"]:
    accuracys = []
    times = []

    results = []
    for rf in result_file:
        with open(rf) as f:
            s = f.read()
            results = json.loads(s)
        
        for r in results["result"]:
            isRuned = (
            (e["block"] == r["option"]["block"]) and
            str(e["double_input"] == r["option"]["double_input"]) and
            str(e["relu_option"] == r["option"]["relu_option"]) and
            (e["concatenate"] == r["option"]["concatenate"]) and
            str(e["reseption"] == r["option"]["reseption"]) and
            str(e["dropout"] == r["option"]["dropout"]) and
            str(e["reseption"] == r["option"]["reseption"]) and
            str(e["wide"]) == r["option"]["wide"] and
            str(e["filters"])== r["option"]["filters"]
            )

            if isRuned == True:
                accuracys.append(float(r["accuracy"]))

                start_time = datetime.datetime.strptime(r["start_time"],'%Y/%m/%d %H:%M:%S')
                end_time = datetime.datetime.strptime(r["end_time"],'%Y/%m/%d %H:%M:%S')
                time = (end_time - start_time).seconds
                times.append(time)

    mean_res = {
        "mean_acc": str(np.mean(accuracys,dtype=float)),
        "var_acc" : str(np.var(accuracys,dtype=float)),
        "mean_time" : str(np.mean(times,dtype=float)),
        "var_time" : str(np.var(times,dtype=float))
    }
    mean_res["option"] = e

    all_result.append(mean_res)

path = "result/"+global_name+"_all.json"
with open(path,mode='w') as w:
    #f.write(json_dump)
    json.dump(all_result,w,indent=4, sort_keys=True, separators=(',',': '))

converter_json.outputcsv_allresult("result/"+global_name+"_all.csv",path)