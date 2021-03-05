import multiprocessing
import time


# %%

def bar():
    for i in range(1000):
        print("Tick")
        time.sleep(1)


if __name__ == '__main__':
    # Start bar as a process
    p = multiprocessing.Process(target=bar)
    p.start()

    # Wait for 10 seconds or until process finishes
    p.join(10)

    # If thread is still active
    if p.is_alive():
        print("process still running... => kill it...")
        p.terminate()  # Terminate - may not work if process is stuck for good
        if p.is_alive():
            p.kill()  # works for sure, no chance for process to finish nicely however

        p.join()

# %%

import signal


# Register an handler for the timeout

def handler(signum, frame):
    raise Exception("Timeout")


def loop_forever():
    import time
    i = 0
    while 1:
        print(f"sec {i}")
        time.sleep(1)
        i += 1


# Register the signal function handler
signal.signal(signal.SIGALRM, handler)

# Define a timeout for your function
signal.alarm(10)

try:
    loop_forever()
except Exception as exc:
    print(exc)
