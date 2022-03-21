import sys
import time
from collections import defaultdict
import tensorflow as tf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tensorflow.python.framework import tensor_util
import matplotlib.pyplot as plt
import collections

plt.rcParams.update({'text.color': "#bdbfc0",
                     'axes.labelcolor': "#bdbfc0",
                     'xtick.color': "#bdbfc0",
                     'ytick.color': "#bdbfc0"})


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, data_path=None):
        tags_set = set()
        data = defaultdict(list)
        first = 0
        last = 0
        flag = True

        for summary in tf.train.summary_iterator(data_path):
            if flag:
                first = summary.wall_time
                flag = False
            last = summary.wall_time
            for v in summary.summary.value:
                t = tensor_util.MakeNdarray(v.tensor)
                data[v.tag].append(t)
                tags_set.add(v.tag)

        start_time = time.ctime(first)
        end_time = time.ctime(last)
        if parent is not None:
            parent.startTimeTraining.setText(start_time)
            parent.endTimeTraining.setText(end_time)

        tags_set = sorted(list(tags_set))
        self.fig = plt.figure()
        self.fig.patch.set_facecolor('white')
        self.fig.patch.set_alpha(0.0)
        self.fig.suptitle('Training charts:')
        for i in range(1, len(data) + 1):
            ax = self.fig.add_subplot(2, 3, i)
            ax.set_title(tags_set[i - 1])
            ax.plot(range(len(data[tags_set[i - 1]])), data[tags_set[i - 1]])
            ax.set_xlabel('Steps', color='#bdbfc0')


        self.fig.tight_layout()
        super().__init__(self.fig)
