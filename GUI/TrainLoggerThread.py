import time
from PyQt5 import QtCore


class TrainLoggerThread(QtCore.QThread):

    log_data = QtCore.pyqtSignal(object)

    def __init__(self):
        QtCore.QThread.__init__(self)

    def run(self):
        path = 'output.txt'
        time.sleep(10)

        while True:
            #self.trainingProcessText.setText(f.read())
            f = open(path, 'r')
            self.log_data.emit(f.read())
            f.close()
            time.sleep(10)
