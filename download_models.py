"""Download the SFace face recognition model, which is too large to store in this repository.

Usage:
    python download_models.py
"""
from pathlib import Path
from urllib.request import urlretrieve

MODELS = Path(__file__).resolve().parent / "models"
FILES = {
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx",
}


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    for name, url in FILES.items():
        target = MODELS / name
        if target.exists() and target.stat().st_size > 1_000_000:
            print(f"{name} is already present ({target.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"Downloading {name} from the OpenCV model zoo ...")
        urlretrieve(url, target)
        print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
