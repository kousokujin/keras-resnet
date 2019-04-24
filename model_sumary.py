from ModelBuilder import ResnetBuilder
from Datasets import mnist_dataset
from keras.utils import plot_model

dataset = mnist_dataset(is_zero_center=False)

option = {
    "relu_option": True,
    "double_input": False,
    "concatenate": "sum",
    "block": "basic_block"
}
model = ResnetBuilder.build_manual(dataset.get_shape(),dataset.get_categorical(),option)
model.summary()
plot_model(model, to_file='model.png')