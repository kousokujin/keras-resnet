import numpy as np
from abc import abstractmethod

from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
import keras.utils.np_utils as kutils

class abstract_dataset:

    @abstractmethod
    def download(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_shape(self):
        raise NotImplementedError()
    @abstractmethod
    def get_name(self):
        raise NotImplementedError()

    def __init__(self):
        (trainX, trainY), (testX, testY) = self.download()

        trainX = trainX.astype('float32')
        trainX /= 255
        testX = testX.astype('float32')
        testX /=255

        trainY = kutils.to_categorical(trainY)
        testY = kutils.to_categorical(testY)

        self.trainX = trainX
        self.testX = testX
        self.trainY = trainY
        self.testY = testY

        #self.init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
        #self.init_shape = (3,32,32)
        self.init_shape = self.get_shape()

class cifer10_datasets(abstract_dataset):

    def download(self):
        return cifar10.load_data()

    def get_shape(self):
        return (3, 32, 32)

    def get_name(self):
        return 'cifer10'

class cifer100_datasets(abstract_dataset):
    
    def download(self):
        return cifar100.load_data(label_mode='fine')

    def get_shape(self):
        return (3, 32, 32)

    def get_name(self):
        return 'cifer100'

class mnist_dataset(abstract_dataset):

    def download(self):
        return mnist.load_data()

    def get_shape(self):
        return (1, 28, 28)

    def get_name(self):
        return 'mnist'


