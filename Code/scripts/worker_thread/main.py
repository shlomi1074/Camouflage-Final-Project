import queue
import time

from worker_thread.worker import WorkerThread

if __name__ == '__main__':
    tasks = [item for item in range(10)]
    q = queue.Queue()
    worker = WorkerThread(q)
    worker.daemon = True
    worker.start()
    for item in tasks:
        q.put(item)
        time.sleep(2)
