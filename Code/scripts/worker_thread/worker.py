import threading
import time


class WorkerThread(threading.Thread):
    def __init__(self, q, group=None, target=None, name=None):
        self.queue = q

        super(WorkerThread, self).__init__(group=group, target=target, name=name)
    # Worker, handles each task
    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                continue
            print("Working on:", item)
            time.sleep(4)
            self.queue.task_done()