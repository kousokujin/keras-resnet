from ModelBuilder import ResnetBuilder

input_shape = (32, 32, 3)
model = ResnetBuilder.build_dualresnet_18(input_shape,10)
model_summary = model.summary()