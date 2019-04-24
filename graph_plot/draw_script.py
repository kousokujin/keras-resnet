from json_graph import json_plot_data
from plot import plot_drawer

paths = ["cifer-10_accuacy.json","cifer-100_accuacy.json","mnist_accuacy.json"]
for p in paths:
    data = json_plot_data(path=p)
    graph = plot_drawer(plot_data=data)
    graph.save()