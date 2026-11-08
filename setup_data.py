
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
from tempfile import TemporaryDirectory
import shutil

# Duong dan bo Project 5 goc cua Berkeley
URL = "https://inst.eecs.berkeley.edu/~cs188/fa25/assets/projects/machinelearning.zip"

# Thu muc data cua project
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REQUIRED_FILES = ["mnist.npz", "lang_id.npz"]

missing = [
    name for name in REQUIRED_FILES
    if not (DATA_DIR / name).is_file()
]

if not missing:
    print("Du lieu da day du.")
else:
    print("Dang tai du lieu tu Berkeley...")

    with TemporaryDirectory() as temp:
        zip_path = Path(temp) / "machinelearning.zip"

        # Tai ZIP goc
        urlretrieve(URL, zip_path)

        with ZipFile(zip_path) as archive:
            for filename in missing:
                matches = [
                    item for item in archive.infolist()
                    if not item.is_dir()
                    and Path(item.filename).name == filename
                    and not item.filename.startswith("__MACOSX/")
                ]

                if len(matches) != 1:
                    raise RuntimeError(
                        f"Khong tim thay duy nhat {filename} trong ZIP"
                    )

                target = DATA_DIR / filename
                temporary = DATA_DIR / (filename + ".part")

                with archive.open(matches[0]) as src:
                    with open(temporary, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                temporary.replace(target)
                print(f"Da tai: {filename}")

    print("Hoan tat chuan bi du lieu!")