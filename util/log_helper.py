import sys
import zipfile
from pathlib import Path

LOG_FOLDER = Path(__file__).parent.parent.parent / "tmp"  # sibling of project folder


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


def get_path_str_without_extension(path: Path):
    assert path.exists()
    assert path.is_file()
    return str((path.parent / f"{path.name.split('.')[0]}").resolve())


def zip_file(file_path: Path) -> Path:
    assert file_path.exists() and file_path.is_file()
    zip_path = Path(f"{get_path_str_without_extension(file_path)}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zip:
        zip.write(file_path)
    return zip_path


def zip_dir(dir_path: Path, file_path: Path):
    assert dir_path.exists() and dir_path.is_dir()
    with zipfile.ZipFile(file_path, "w") as zip:
        for entry in dir_path.rglob("*"):
            zip.write(entry, entry.relative_to(dir_path))
