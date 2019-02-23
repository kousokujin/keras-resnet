import numpy as np

from keras.datasets import cifar10
import keras.callbacks as callbacks
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
        self.nb_epoch = 30
        self.img_rows, self.img_cols = 32, 32

    def run_model(self,model):
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
        self.history = model.fit(self.trainX,self.trainY,batch_size=self.batch_size,nb_epoch=self.nb_epoch,verbose=1,validation_split=0.1)
        self.model = model

    def evalute_model(self):
        self.score=self.model.evaluate(self.testX,self.testY,verbose=0)
        print('Test loss:',self.score[0])
        print('Test accuracy:',self.score[1])

    def __init__(self,name):
        self.name = name

class datasets:
    
    def __init__(self):

        #データセット
        (trainX, trainY), (testX, testY) = cifar10.load_data()

        trainX = trainX.astype('float32')
        #trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
        trainX /= 255
        testX = testX.astype('float32')
        #testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))
        testX /=255

        trainY = kutils.to_categorical(trainY)
        testY = kutils.to_categorical(testY)

        self.trainX = trainX
        self.testX = testX
        self.trainY = trainY
        self.testY = testY

        #self.init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
        self.init_shape = (3,32,32)