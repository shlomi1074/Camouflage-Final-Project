import time

import tensorflow as tf
from tensorflow.python.framework import tensor_util
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d
from collections import defaultdict

tags_set = set()
data = defaultdict(list)

first = 0
last = 0
flag = True
for summary in tf.train.summary_iterator(r"E:\FinalProject\backup_files\events.out.tfevents.1646244711.SHLOMI-PC.18044.0.v2"):
    print(summary)
    # for v in summary.wall_time:
    #     print(v)
        #t = tensor_util.MakeNdarray(v.tensor)

# plt.plot(range(1, conf_loss_steps), conf_loss_data, linewidth=1.5, linestyle='dotted')
# plt.xlabel("Steps")
# plt.ylabel("Confidence loss")
# plt.show()
#fig, ax = plt.subplots()
#_ = ax.plot(range(1, conf_loss_steps), prob_loss_data)
#plt.xlabel("Steps")
#plt.ylabel("Probability loss")
#xnew = np.linspace(2, prob_loss_steps, 200)
# z = np.polyfit(range(1, conf_loss_steps), prob_loss_data, 1)
# p = np.poly1d(z)
#spl = make_interp_spline(range(1, conf_loss_steps), prob_loss_data, k=3)  # type: BSpline
#ysmoothed = gaussian_filter1d(prob_loss_data, sigma=3)

#power_smooth = spl(xnew)

#plt.plot(xnew, power_smooth)
#plt.plot(range(1, conf_loss_steps), ysmoothed)

#plt.show()