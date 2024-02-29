import signal
import time


# Register an handler for the timeout

def handler(signum, frame):
    raise Exception("Timeout")


def loop_forever():
    i = 0
    while True:
        print(f"{i} seconds passed")
        time.sleep(1)
        i += 1


if __name__ == '__main__':
    # Register the signal function handler
    signal.signal(signal.SIGALRM, handler)

    # Define a timeout for your function
    signal.alarm(10)

    try:
        loop_forever()
    except Exception as exc:
        print(exc)
