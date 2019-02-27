import numpy as np
import sklearn.metrics as metrics
from ModelBuilder import ResnetBuilder
from plot_result import (plot_acc_history,plot_loss_history,outputfile_evalute,get_time)
from ResNetTester_cls import ResNetTester
from Datasets import (cifer10_datasets,cifer100_datasets,mnist_dataset)

batch_sizes = [32,64,128,256]
validation_split = [0.1,0.2,0.3]
method = [
    'ResNet18',
    'Invert_ResNet18',
    'Dual_ReLU_ResNet18',
    'Dual_ReLU_Concatenate_ResNet18'
]

#データセット
datas = [cifer10_datasets,cifer100_datasets,mnist_dataset]

output_log_file = get_time() + '_' + 'result.txt'

testers = []

for b in batch_sizes:
    testers_x = []
    for v in validation_split:
        testers_y = []
        for m in method:
            testers_z = []
            for d in datas:
                dataset_name = d.get_name()
                testname = m+'batchsize_'+str(b)+'validdation_'+str(v)+dataset_name
                test = ResNetTester(testname)
                test.setDataset(d)
                test.batch_size = b
                test.validation_split = v
                test.nb_epoch = 10

                if m == 'ResNet18':
                    model = ResnetBuilder.build_resnet_18(d.get_shape(),10)
                elif m == 'Invert_ResNet18':
                    model = ResnetBuilder.build_invert_relu_resnet_18(d.get_shape(),10)
                elif m == 'Dual_ReLU_ResNet18':
                    model = ResnetBuilder.build_dualresnet_18(d.get_shape(),10)
                elif m == 'Dual_ReLU_Concatenate_ResNet18':
                    model = ResnetBuilder.build_concatenate_dualresnet_18(d.get_shape(),10)

                test.run_model(model)
                test.evalute_model()
                output_log_file(test.score,test.name,path=output_log_file)
                testers_z.append(test)
            testers_y.append(testers_z)
        testers_x.append(testers_y)
    testers.append(testers_x)

historys = []
legends = []
            
        