import threading
import time
from threading import Thread


def some_process():
    n_steps = 10
    for i in range(n_steps):
        print(f"Thread {threading.current_thread().name} - progress {i / n_steps * 100:3.0f} %")
        time.sleep(1)


def print_thread_info(t: Thread):
    print(f"Current thread name: {t.name}")
    print(f"Thread identifier: {threading.get_ident()}")
    print(f"Thread system identifier: {threading.get_native_id()}")


print(f"Number of threads: {threading.active_count()}")
print_thread_info(threading.current_thread())

# Run a thread (Variant 1: pass a callable object to the Thread constructor)
t1 = Thread(target=some_process, name="Thread 1")
t1.start()


# Run a thread (Variant 2: overwrite run() method)
class MyThread(Thread):
    def __init__(self):
        super().__init__(name="Thread 2")
        self.daemon = True

    def run(self):
        some_process()


t2 = MyThread()
time.sleep(3)
t2.start()

# We wait for Thread 1.
# The program will then terminate, since Thread 2 is a daemon
t1.join()
print("Thread 1 finished")
