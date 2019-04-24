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

def model_build(model_name,input_shape,categorical):
    double_option = {"relu_option": False, "double_input": True}
    invert_option = {"relu_option":True, "double_input": True}

    if model_name == 'ResNet18':
        model = ResnetBuilder.build_resnet_18(input_shape,categorical)
    elif model_name == 'Invert_ResNet18':
        model = ResnetBuilder.build_invert_relu_resnet_18(input_shape,categorical)
    elif model_name == 'Dual_ReLU_ResNet18':
        model = ResnetBuilder.build_dualresnet_18(input_shape,categorical)
    elif model_name == 'Dual_ReLU_Concatenate_ResNet18':
        model = ResnetBuilder.build_concatenate_dualresnet_18(input_shape,categorical)
    elif model_name == 'ResNet18-DoubleInput':
        model = ResnetBuilder.build_resnet_18(input_shape,categorical,option = double_option)
    elif model_name == 'Invert_ResNet18-DoubleInput':
        model = ResnetBuilder.build_invert_relu_resnet_18(input_shape,categorical, option=invert_option)
    elif model_name == 'Dual_ReLU_ResNet18-DoubleInput':
        model = ResnetBuilder.build_dualresnet_18(input_shape,categorical,option = double_option)
    elif model_name == 'Dual_ReLU_Concatenate_ResNet18-DoubleInput':
        model = ResnetBuilder.build_concatenate_dualresnet_18(input_shape,categorical,option = double_option)

    return model

def isExist(test_dec,json_path):
    with open(json_path) as f:
        s = f.read()
        json_data = json.loads(s)
    data = json_data["result"]
    exist = False

    for d in data:
        exist_temp = True
        for dec_key in test_dec.keys():
            if include_dict(d,dec_key,test_dec[dec_key]) == False:
                exist_temp = False
        
        if exist_temp == True:
            exist = True
    
    return exist
            



global_testname = "20190329_test.json"
batch_size=[32,128]
#batch_size=[512,1024]
epochs = [10,30,50]
#epochs = [1,2]
methods = [
    'ResNet18',
    'Dual_ReLU_Concatenate_ResNet18',
    'ResNet18-DoubleInput',
    'Dual_ReLU_Concatenate_ResNet18-DoubleInput'
]
#methods = ['ResNet18','ResNet18-DoubleInput']
datas = [
    cifer10_datasets(is_zero_center=True),
    cifer100_datasets(is_zero_center=True),
    mnist_dataset(is_zero_center=True),
]
#datas = [mnist_dataset(is_zero_center=True)]
'''
test_dec={
    "batch_size": "32",
    "dataset": "cifer100",
    "epoch": "10",
    "method": "ResNet18"
}
exist = isExist(test_dec,'result/'+global_testname)
'''

for d in datas:
    for b in batch_size:
        for e in epochs:
            historys = []
            legends = []
            for m in methods:
                test_dec={
                    "batch_size": str(b),
                    "dataset": d.get_name(),
                    "epoch": str(e),
                    "method": m
                }
                debug = isExist(test_dec, 'result/'+global_testname)
                if isExist(test_dec, 'result/'+global_testname) == False:
                    dataset_name = d.get_name()
                    testname = m+'_batchsize_'+str(b)+'_epoch_'+str(e)+'_'+dataset_name
                    test = ResNetTester(testname)
                    test.setDataset(d)
                    test.batch_size = b
                    test.nb_epoch = e
                    model = model_build(m,d.get_shape(),d.get_categorical())
                    test.run_model(model,m)
                    test.evalute_model()

                    historys.append(test.history)
                    legends.append(test.name)
                    json_write(test,'result/'+global_testname)

            #path= 'result/'+testname+'.png'
            #plot_acc_history(historys,legends,path=path)
            #historys.clear()
            #legends.clear()
    
        graph_title = d.get_name()+'_batch_size_'+str(b)+'_accuracy'
        graph_config = {
            "title": graph_title,
            "file_name": 'result/'+graph_title+'.png',
            "json_path": 'result/'+global_testname,
            "fixed_datas": {"batch_size":str(b),"dataset": d.get_name()},
            "x1": "epoch",
            "x2": "method",
            "y": "accuracy",
            "auto": "true"
        }
        json_bar2_graph(graph_config,"result/"+graph_title+'.json')
        plot_data = json_plot_data("result/"+graph_title+'.json')
        plot_draw = plot_drawer(plot_data)
        plot_draw.save()

