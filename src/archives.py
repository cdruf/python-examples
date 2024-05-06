import zipfile
from pathlib import Path


def get_path_str_without_extension(path: Path):
    assert path.exists()
    assert path.is_file()
    return str((path.parent / f"{path.name.split('.')[0]}").resolve())


# Create a folder
folder_path = Path(f'./tmp/')
if not folder_path.exists():
    folder_path.mkdir()
assert folder_path.exists()
assert folder_path.is_dir()

# Create a file
file_path = Path(folder_path / 'tmp.txt')
with open(file_path, 'w') as f:
    f.write('hello')
assert file_path.exists()
assert file_path.is_file()

# Zip file
with zipfile.ZipFile(f"{get_path_str_without_extension(file_path)}.zip", 'w') as f:
    f.write(file_path)

with zipfile.ZipFile('tmp_folder.zip', 'w') as f:
    for file in folder_path.iterdir():
        f.write(file)

print("")
