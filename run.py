import numpy as np
import sklearn.metrics as metrics
from ModelBuilder import ResnetBuilder
from plot_result import (plot_acc_history,plot_loss_history,outputfile_evalute)
from ResNetTester_cls import ResNetTester

ResNetTest = ResNetTester('ResNet18')
ResNetModel = ResnetBuilder.build_resnet_18(ResNetTest.init_shape,10)
ResNetTest.run_model(ResNetModel)
ResNetTest.evalute_model()

Inv_ResNetTest = ResNetTester('INV_ResNet18')
INV_ResNetModel = ResnetBuilder.build_dualresnet_18(Inv_ResNetTest.init_shape,10)
Inv_ResNetTest.run_model(INV_ResNetModel)
Inv_ResNetTest.evalute_model()

outputfile_evalute(ResNetTest.score,ResNetTest.name)
outputfile_evalute(Inv_ResNetTest.score,Inv_ResNetTest.name)
historys = [ResNetTest.history,Inv_ResNetTest.history]
plot_acc_history(historys,legend=['ResNet_acc','INV_ResNet_acc'])
plot_loss_history(historys,legend=['ResNet_loss','INV_ResNet_loss'])