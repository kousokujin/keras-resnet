from ModelBuilder import ResnetBuilder
from Datasets import (cifer10_datasets,cifer100_datasets,mnist_dataset)
from keras.utils import plot_model

dataset = cifer10_datasets(is_zero_center=False)

option = {
    "relu_option": False,
    "double_input": True,
    "concatenate": "full_concanate",
    "block": "double_bottleneck",
    "reseption" :[3,4,6,3]
}
model = ResnetBuilder.build_manual(dataset.get_shape(),dataset.get_categorical(),option)
model.summary()
#plot_model(model, to_file='model.png')