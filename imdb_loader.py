import os
import tarfile
import urllib.request
from pathlib import Path

def download_imdb(data_dir="data/imdb_raw"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = data_dir / "aclImdb_v1.tar.gz"

    if not tar_path.exists():
        print("Downloading IMDb dataset (~80MB)...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")

    extract_dir = data_dir / "aclImdb"
    if not extract_dir.exists():
        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")

    return extract_dir  # e.g. data/imdb_raw/aclImdb

def load_imdb_texts_labels(base_dir, split="train"):
    """
    base_dir/train/pos , base_dir/train/neg , etc.
    label 1 = pos, 0 = neg
    """
    base_dir = Path(base_dir)
    texts = []
    labels = []

    for label_name, label_val in [("pos",1), ("neg",0)]:
        folder = base_dir / split / label_name
        for fname in folder.iterdir():
            if fname.suffix == ".txt":
                txt = fname.read_text(encoding="utf-8", errors="ignore")
                texts.append(txt)
                labels.append(label_val)

    return texts, labels

def get_imdb_dataset():
    """
    Returns imdb_data dict matching preprocess.prepare_imdb_dataloaders() expectations.
    """
    root = download_imdb()
    train_texts, train_labels = load_imdb_texts_labels(root, split="train")
    test_texts, test_labels   = load_imdb_texts_labels(root, split="test")

    imdb_data = {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "test_texts": test_texts,
        "test_labels": test_labels
    }
    return imdb_data
