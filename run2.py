import numpy as np
import sklearn.metrics as metrics
from ModelBuilder import ResnetBuilder
from plot_result import (plot_acc_history,plot_loss_history,outputfile_evalute,get_time,output_text)
from ResNetTester_cls import ResNetTester
from Datasets import (cifer10_datasets,cifer100_datasets,mnist_dataset)

#batch_sizes = [32,64,128]
batch_sizes = [128]
#validation_split = [0.1,0.2,0.3]
methods = [
    'ResNet18',
    'Invert_ResNet18',
    'Dual_ReLU_ResNet18',
    'Dual_ReLU_Concatenate_ResNet18'
]

#データセット
#datas = [cifer10_datasets(),cifer100_datasets(),mnist_dataset()]
datas = [mnist_dataset()]
output_log_file = get_time() + '_' + 'result.txt'
#testers = []

for d in datas:
    output_text(d.get_name(),output_log_file)
    for b in batch_sizes:
        output_text('batch_size:'+str(b),output_log_file)
        historys = []
        legends = []
        for m in methods:
            dataset_name = d.get_name()
            testname = m+'_batchsize_'+str(b)+dataset_name
            test = ResNetTester(testname)
            test.setDataset(d)
            test.batch_size = b
            test.nb_epoch = 20

            if m == 'ResNet18':
                model = ResnetBuilder.build_resnet_18(d.get_shape(),d.get_categorical())
            elif m == 'Invert_ResNet18':
                model = ResnetBuilder.build_invert_relu_resnet_18(d.get_shape(),d.get_categorical())
            elif m == 'Dual_ReLU_ResNet18':
                model = ResnetBuilder.build_dualresnet_18(d.get_shape(),d.get_categorical())
            elif m == 'Dual_ReLU_Concatenate_ResNet18':
                model = ResnetBuilder.build_concatenate_dualresnet_18(d.get_shape(),d.get_categorical())

            test.run_model(model)
            test.evalute_model()
            outputfile_evalute(test.score,test.name,path=output_log_file)
            
            historys.append(test.history)
            legends.append(test.name)
        
        path= 'result/'+testname+'.png'
        plot_acc_history(historys,legends,path=path)
        historys.clear()
        legends.clear()
            
        