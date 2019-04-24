import json
from collections import OrderedDict
import pprint
import os
import numpy as np

class json_plot_data:

    def __init__(self,path):
        name = os.path.dirname(os.path.abspath(__file__))
        joinpath = os.path.join(name,path)
        datapath = os.path.normpath(joinpath)

        s = ""
        with open(path) as f:
            s = f.read()

        self.dict_json= json.loads(s)
    
    def get_axis_name(self):
        x_name = self.dict_json["x_name"]
        y_name = self.dict_json["y_name"]

        return (x_name,y_name)

    def get_type(self):
        return self.dict_json["type"]

    def get_x(self):
        return self.dict_json["x"]

    def get_y(self):
        return self.dict_json["y"]

    def get_title(self):
        return self.dict_json["title"]

    def get_file_name(self):
        return self.dict_json["file_name"]
    
    def get_labels(self):
        return self.dict_json["labels"]

    def get_minmax(self):
        
        max = 1.0
        min = 0.0
        if not self.dict_json.get("max") is None:
            max = float(self.dict_json["max"])
        
        if not self.dict_json.get("min") is None:
            min = float(self.dict_json["min"])

        if not self.dict_json.get("auto") is None:
            if self.dict_json["auto"] == 'true':
                (max, min) = self.auto_range()


        return (max,min)
    
    def auto_range(self):
        y = np.array(self.get_y(), dtype=float)
        max = np.max(y)
        min = np.min(y)
        range = max - min

        max += range*0.25
        min -= range*0.25

        return (max,min)