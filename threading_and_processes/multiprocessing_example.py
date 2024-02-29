"""
The multiprocessing package uses subprocesses instead of threads.
Therefore, it allows to terminate processes from the outside.
This is useful when cannot implement any hooks in a thread to abort it.
"""
import multiprocessing
import time


def some_process(n, name):
    for i in range(n):
        print(f"Process {name} --- {i:3.0f} %")
        time.sleep(1)


if __name__ == '__main__':
    # Start process that takes a while
    p = multiprocessing.Process(target=some_process, args=(100, "1"))
    p.start()

    # Wait 5 seconds (OR until process finishes)
    p.join(5)

    if p.is_alive():  # => the thread is still active
        print("Process still running => terminate it")
        p.terminate()  # may not work if process is stuck for good
        p.join()  # must be called after terminate or the process remains alive
        if p.is_alive():
            print("Process could not be terminated => kill it")
            p.kill()  # works for sure, no chance for process to finish nicely however
            p.join()

    print(f"Process alive = {p.is_alive()}")
