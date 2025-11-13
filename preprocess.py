
import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from utils import set_seed
import numpy as np

class IMDBDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)  # remove punctuation/special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    # simple whitespace tokenizer
    return text.split()

def build_vocab(tokenized_texts, vocab_size=10000, min_freq=1):
    counter = Counter()
    for toks in tokenized_texts:
        counter.update(toks)
    # reserve 0 for PAD, 1 for OOV
    most_common = [w for (w, c) in counter.items() if c >= min_freq]
    most_common = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    most_common = [w for (w, _) in most_common[:vocab_size-2]]
    word2idx = {"<PAD>":0, "<OOV>":1}
    for i,w in enumerate(most_common, start=2):
        word2idx[w] = i
    return word2idx

def text_to_ids(tokens, word2idx):
    return [word2idx.get(t, 1) for t in tokens]  # 1 is OOV

def pad_or_truncate(seq, max_len):
    if len(seq) < max_len:
        return seq + [0]*(max_len - len(seq))
    else:
        return seq[:max_len]

def prepare_imdb_dataloaders(
        imdb_data,
        seq_len=50,
        batch_size=32,
        vocab_size=10000,
        seed=42,
        shuffle_train=True):
    """
    imdb_data is expected to be:
    {
      "train_texts": [...],
      "train_labels": [... 0/1 ...],
      "test_texts": [...],
      "test_labels": [...]
    }
    We'll:
      - clean
      - tokenize
      - build vocab on train only
      - convert and pad both train/test
    """
    set_seed(seed)

    train_clean = [clean_text(t) for t in imdb_data["train_texts"]]
    test_clean  = [clean_text(t) for t in imdb_data["test_texts"]]

    train_tokens = [tokenize(t) for t in train_clean]
    test_tokens  = [tokenize(t) for t in test_clean]

    word2idx = build_vocab(train_tokens, vocab_size=vocab_size)

    train_ids = [text_to_ids(toks, word2idx) for toks in train_tokens]
    test_ids  = [text_to_ids(toks, word2idx) for toks in test_tokens]

    train_pad = [pad_or_truncate(seq, seq_len) for seq in train_ids]
    test_pad  = [pad_or_truncate(seq, seq_len) for seq in test_ids]

    train_dataset = IMDBDataset(train_pad, imdb_data["train_labels"])
    test_dataset  = IMDBDataset(test_pad, imdb_data["test_labels"])

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle_train)
    test_loader  = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    return train_loader, test_loader, word2idx
