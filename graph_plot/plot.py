from json_graph import json_plot_data

import numpy as np
import matplotlib.pyplot as plt

class plot_drawer:

    def __init__(self,plot_data):
        self.data = plot_data
        self.colors = ["red","royalblue","orange","lime","sienna","deeppink"]

    def draw(self):
        if self.data.get_type() == "bar":
            self.bar()
        if self.data.get_type() == "plot":
            self.line()

    def show(self):
        self.draw()
        plt.show()

    def save(self):
        self.draw()
        plt.savefig(self.data.get_file_name())


    def bar(self):
        plt.clf()
        y = self.data.get_y()
        np_y = np.array(y)
        y_count = 0

        for i in np_y:
            count = len(i)
            if(y_count < count):
                y_count = count
        
        labels = self.data.get_x()
        axis_label = self.data.get_labels()
        width = 0.1
        now_width = 0

        (max,min) = self.data.get_minmax()
        plt.ylim(min,max)

        for r ,i in  enumerate(np_y):
            left = np.arange(y_count,dtype=float)
            left += now_width
            y_float = np.array(i,dtype=float)
            self.plot = plt.bar(left,y_float,color=self.colors[r], width=width, align='center',label=axis_label[r])
            now_width += width

        x_label_pos = np.arange(y_count,dtype=float)
        shift = (width*(len(y)-1))/2
        plt.xticks(x_label_pos + shift, labels)

        (x_title,y_title) = self.data.get_axis_name()
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.title(self.data.get_title())

        plt.legend()

    def line(self):
        plt.clf()
        y = self.data.get_y()
        x = self.data.get_x()
        axis_label = self.data.get_labels()
        np_y = np.array(y)

        for r,i in enumerate(np_y):
            plt.plot(x,i,color=self.colors[r],label=axis_label[r])
        
        (x_title,y_title) = self.data.get_axis_name()
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.title(self.data.get_title())
        plt.legend()


