import sys
import time
from collections import defaultdict
import tensorflow as tf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tensorflow.python.framework import tensor_util
import matplotlib.pyplot as plt
import collections

plt.rcParams.update({'text.color': "#283618",
                     'axes.labelcolor': "#283618",
                     'xtick.color': "#283618",
                     'ytick.color': "#283618"})


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, data_path=None):
        plt.style.use('bmh')

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
                if 'static_view' in v.tag or 'gradients' in v.tag or 'image' in v.tag:
                    continue
                if 'gan' in v.tag or 'ae_loss' in v.tag or 'g_loss':
                    data[v.tag].append(v.simple_value)
                else:
                    t = tensor_util.MakeNdarray(v.tensor)
                    data[v.tag].append(t)
                tags_set.add(v.tag)

        start_time = time.ctime(first)
        end_time = time.ctime(last)
        if parent is not None:
            parent.startTimeTraining.setText(start_time)
            parent.endTimeTraining.setText(end_time)

        if 'lr' in tags_set:
            parent.resultsToggleButton.setVisible(True)
        else:
            parent.resultsToggleButton.setVisible(False)


        tags_set = sorted(list(tags_set))
        self.fig = plt.figure()
        self.fig.patch.set_facecolor('white')
        self.fig.patch.set_alpha(0.0)
        #self.fig.suptitle('Training charts:')
        temp = 0
        for i in range(1, len(data) + 1):
            if 'x2' in tags_set[i - 1]:
                temp += 1
                continue
            ax = self.fig.add_subplot(2, 3, i - temp)
            ax.set_title(tags_set[i - 1])
            ax.plot(range(len(data[tags_set[i - 1]])), data[tags_set[i - 1]])
            # ax.set_xlabel('Steps', color='#bdbfc0')

        parent.training_data = data
        self.fig.tight_layout()
        super().__init__(self.fig)
