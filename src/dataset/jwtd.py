import shutil
import tarfile
import urllib.request
from pathlib import Path

JWTD_URL = "https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JWTD/jwtd.tar.gz"
DATA_DIR = Path("data/jwtd")


def prepare_jwtd():
    target_file = DATA_DIR / "test.jsonl"
    alt_target_file = DATA_DIR / "jwtd" / "test.jsonl"

    if target_file.exists() or alt_target_file.exists():
        return

    print(f"JWTD not found. Downloading from {JWTD_URL}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "jwtd.tar.gz"

    try:
        urllib.request.urlretrieve(JWTD_URL, tar_path)
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        if tar_path.exists():
            tar_path.unlink()
        raise

    print("Extracting JWTD...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)

        subdir = DATA_DIR / "jwtd"
        if subdir.exists() and subdir.is_dir():
            for item in subdir.iterdir():
                destination = DATA_DIR / item.name
                if destination.exists():
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                shutil.move(str(item), str(DATA_DIR))
            subdir.rmdir()

    except Exception as e:
        print(f"Failed to extract dataset: {e}")
        raise
    finally:
        if tar_path.exists():
            tar_path.unlink()

    print("JWTD preparation complete.")
