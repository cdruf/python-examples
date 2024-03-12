import sys
from pathlib import Path


class Logger:

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
    path = Path('./log.txt')
    Logger(path)
    print("Hi")
