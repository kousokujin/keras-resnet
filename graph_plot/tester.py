from json_graph import json_plot_data
from plot import plot_drawer

d = json_plot_data(path="cifer-10_accuacy.json")
graph = plot_drawer(plot_data=d)
graph.show()
graph.save()