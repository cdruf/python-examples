import sys
from datetime import datetime
from pathlib import Path


class MyLogger:

    def __init__(self, log_file_path):
        self.file = open(log_file_path, 'a')
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


if __name__ == '__main__':
    path = Path('../test.log')
    my_logger = MyLogger(path)
    print(f"{datetime.now()}: Test log entry")
