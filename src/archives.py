import zipfile
from pathlib import Path


def get_path_str_without_extension(path: Path):
    assert path.exists()
    assert path.is_file()
    return str((path.parent / f"{path.name.split('.')[0]}").resolve())


# Create a folder
my_folder = Path(f'./tmp/')
if not my_folder.exists():
    my_folder.mkdir()
assert my_folder.exists() and my_folder.is_dir()

# Create a file
my_file = Path(my_folder / 'tmp.txt')
with open(my_file, 'w') as f:
    f.write('hello')
assert my_file.exists() and my_file.is_file()


def zip_file(file_path: Path) -> Path:
    assert file_path.exists() and file_path.is_file()
    zip_path = Path(f"{get_path_str_without_extension(file_path)}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zip:
        zip.write(file_path)
    return zip_path


archive_path = zip_file(my_file)
assert archive_path.exists() and archive_path.is_file()
print(archive_path)


def zip_dir(dir_path: Path, file_path: Path):
    assert dir_path.exists() and dir_path.is_dir()
    with zipfile.ZipFile(file_path, "w") as zip:
        for entry in dir_path.rglob("*"):
            zip.write(entry, entry.relative_to(dir_path))


zip_dir(my_folder, my_folder.parent / f"{my_folder.name}.zip")
