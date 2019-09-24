#from ModelBuilder import ResnetBuilder
from model_module import ModelBuilder
from Datasets import (cifer10_datasets,cifer100_datasets,mnist_dataset)
from keras.utils import plot_model

dataset = cifer10_datasets(is_zero_center=False)

option = {
    "relu_option": False,
    "double_input": False,
    "concatenate": "full_concanate",
    "block": "double_basic",
    "reseption" :[2,2,2],
    "dropout": 0,
    "wide":True,
    "filters": 64
}
model = ModelBuilder.ResnetBuilder.build_manual(dataset.get_shape(),dataset.get_categorical(),option)
model.summary()
plot_model(model, to_file='model.png')