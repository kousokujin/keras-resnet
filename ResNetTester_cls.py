import numpy as np

import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from ModelBuilder import ResnetBuilder
from keras import backend as K

class ResNetTester:

    def setDataset(self,dataset):
        self.trainX = dataset.trainX
        self.trainY = dataset.trainY
        self.testX = dataset.testX
        self.testY = dataset.testY

        #バッチサイズなど
        self.batch_size = 128
        self.nb_epoch = 40
        self.validation_split = 0.1
        #self.img_rows, self.img_cols = 32, 32

    def run_model(self,model):
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
        self.history = model.fit(self.trainX,self.trainY,batch_size=self.batch_size,nb_epoch=self.nb_epoch,verbose=1,validation_split=self.validation_split)
        self.model = model

    def evalute_model(self):
        self.score=self.model.evaluate(self.testX,self.testY,verbose=0)
        print('Test loss:',self.score[0])
        print('Test accuracy:',self.score[1])

    def __init__(self,name):
        self.name = name