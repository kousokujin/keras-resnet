'''
    それぞれのネットワークの評価コード
'''

import numpy as np
import sklearn.metrics as metrics
from ModelBuilder import ResnetBuilder
from plot_result import (plot_acc_history,plot_loss_history,outputfile_evalute,get_time)
from ResNetTester_cls import ResNetTester
from Datasets import cifer10_datasets

data = cifer10_datasets()
Testers = [
    ResNetTester('ResNet18'),
    ResNetTester('Invert_ResNet18'),
    ResNetTester('Dual_ReLU_ResNet18'),
    ResNetTester('Dual_ReLU_Concatenate_ResNet18')
]
output_log_file = get_time() + '_' + 'result.txt'

for index, test in enumerate(Testers):
    test.setDataset(data)

    if index == 0:
        model = ResnetBuilder.build_resnet_18(data.init_shape,10)
    elif index == 1:
        model = ResnetBuilder.build_invert_relu_resnet_18(data.init_shape,10)
    elif index == 2:
        model = ResnetBuilder.build_dualresnet_18(data.init_shape,10)
    elif index == 3:
        model = ResnetBuilder.build_concatenate_dualresnet_18(data.init_shape,10)
    
    test.run_model(model)
    test.evalute_model()
    outputfile_evalute(test.score,test.name,path=output_log_file)

historys = []
legends = []
for test in Testers:
    historys.append(test.history)
    legends.append(test.name)


plot_acc_history(historys,legend=legends)