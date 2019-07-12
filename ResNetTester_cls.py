import numpy as np

import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from ModelBuilder import ResnetBuilder
from keras import backend as K
from datetime import datetime

#from json_writer import json_write

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

    def evalute_model(self):
        self.score=self.model.evaluate(self.testX,self.testY,verbose=0)
        print('Test loss:',self.score[0])
        print('Test accuracy:',self.score[1])
        self.accuracy = self.score[1]
        self.loss = self.score[0]

    def run(self,model,global_name):
        self.run_model(model)
        self.evalute_model()
        #json_write(self,'result/'+global_name)

    def __init__(self,option):
        self.option = option