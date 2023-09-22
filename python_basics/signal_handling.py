import signal

valid_signals = signal.valid_signals()
print(valid_signals)


def handler(signum, frame):
    signame = signal.Signals(signum).name
    print(f'Signal handler called with signal {signame} ({signum})')
    raise RuntimeError("Signal error")


# Set the signal handler and a 5-second alarm
signal.signal(signal.SIGALRM, handler)
signal.alarm(3)

while True:
    pass

signal.alarm(0)  # Disable the alarm
