import numpy as np
import os
import json

import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
#from ModelBuilder import ResnetBuilder
from model_module import ModelBuilder
from keras import backend as K
from datetime import datetime

#from json_writer import json_write
from tools import json_writer

class ResNetTester:

    def setDataset(self,dataset):
        self.trainX = dataset.trainX
        self.trainY = dataset.trainY
        self.testX = dataset.testX
        self.testY = dataset.testY
        
        test_data_shape = np.shape(self.testX)
        train_data_shape = np.shape(self.trainX)
        self.dataset_name = dataset.get_name()
        self.test_data = test_data_shape[0]
        self.train_data = train_data_shape[0]

        #バッチサイズなど
        self.batch_size = 128
        self.nb_epoch = 40
        self.validation_split = 0.1
        #self.img_rows, self.img_cols = 32, 32

    def run_model(self,model):
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        self.start_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.history = model.fit(
            self.trainX,
            self.trainY,
            batch_size=self.batch_size,
            nb_epoch=self.nb_epoch,
            validation_data=(self.testX,self.testY),
            shuffle= True)
        self.end_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.model = model
    
    def run_model_augmentation(self,model):
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        self.start_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False
        )  # randomly flip image

        datagen.fit(self.trainX)
        model.fit_generator(datagen.flow(self.trainX, self.trainY, batch_size=self.batch_size),
            steps_per_epoch=self.trainX.shape[0] ,
            validation_data=(self.testX,self.testY),
            epochs=self.nb_epoch, verbose=1, max_q_size=100
        )

        self.model = model



    def evalute_model(self):
        self.score=self.model.evaluate(self.testX,self.testY,verbose=0)
        print('Test loss:',self.score[0])
        print('Test accuracy:',self.score[1])
        self.accuracy = self.score[1]
        self.loss = self.score[0]

    def run(self,model,global_name,argment=False):
        
        if argment == False:
            self.run_model(model)
        else:
            self.run_model_augmentation(model)
        self.evalute_model()
        self.model_save(global_name)
        json_writer.json_write(self,'result/'+global_name+'.json')
    
    def model_save(self,global_name):
        print('save models')
        directory_name = 'result/'+global_name+'_models/'
        
        method = self.option["block"]
        concanate = self.option["concatenate"]
        double_input = str(self.option["double_input"])
        dropout = str(self.option["dropout"])

        filename =  directory_name+self.dataset_name+'_'+method+'_'+concanate+'_'+double_input+'_'+dropout+'.h5'

        if os.path.isdir(directory_name) == False:
            os.makedirs(directory_name)

        self.model.save(filename)

        print('saved model!!('+filename+')')

    def __init__(self,option):
        self.option = option


def make(option,dataset,batch_size,epochs,split):
    tester = ResNetTester(option = option)
    tester.setDataset(dataset)
    tester.batch_size = batch_size
    tester.nb_epoch = epochs
    tester.validation_split = split
    model = ModelBuilder.ResnetBuilder.build_manual(input_shape=dataset.get_shape(),num_outputs=dataset.get_categorical(), option = option)
    return model,tester

def run(model,tester,global_name,argment=False):
    tester.run(model,global_name,argment=argment)

def check_run(option,json_result):

    if not os.path.exists(json_result):
        return False
    
    with open(json_result) as f:
        s = f.read()
        json_dict = json.loads(s)

    str_option = {
        "relu_option": str(option["relu_option"]),
        "double_input": str(option["double_input"]),
        "block": str(option["block"]),
        "concatenate": str(option["concatenate"]),
        "reseption" : str(option["reseption"]),
        "dropout": str(option["dropout"])
    }

    for data in json_dict["result"]:

        isRuned = (
            (str_option["block"] == data["option"]["block"]) and
            (str_option["double_input"] == data["option"]["double_input"]) and
            (str_option["relu_option"] == data["option"]["relu_option"]) and
            (str_option["concatenate"] == data["option"]["concatenate"]) and
            (str_option["reseption"] == data["option"]["reseption"]) and
            (str_option["dropout"] == data["option"]["dropout"])
        )

        if isRuned == True:
            return True
    
    return False
        