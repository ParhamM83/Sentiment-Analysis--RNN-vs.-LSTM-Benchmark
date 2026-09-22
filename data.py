import re
import torch
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
PAD_IDX = 0
UNK_IDX = 1


class Vocabulary:
    def __init__(self):
        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.counts = Counter()

    def build_vocab(self, texts, max_size=20000, min_freq=2):
        for text in texts:
            tokens = text.split()
            self.counts.update(tokens)

        filtered_counts = Counter({w: c for w, c in self.counts.items() if c >= min_freq})
        most_common = filtered_counts.most_common(max_size)

        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, word):
        return self.word2idx.get(word, UNK_IDX)

    def __contains__(self, word):
        return word in self.word2idx


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = text.split()
        ids = [self.vocab.word2idx.get(token, UNK_IDX) for token in tokens]
        if not ids:
            ids = [UNK_IDX]

        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)  # Remove HTML tags (e.g. <br />, <br>)
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # Keep alphanumeric and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_reviews(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    texts, labels = [], []
    for label, subdir in enumerate(["neg", "pos"]):
        folder = data_dir / subdir
        if not folder.exists():
            continue
        files = [f for f in folder.iterdir() if f.suffix == ".txt"]
        for file in tqdm(files, desc=f"Loading {data_dir.name}/{subdir}", leave=False):
            texts.append(clean_text(file.read_text(encoding='utf-8')))
            labels.append(label)
    return texts, labels


def collate(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([text.size(0) for text in texts], dtype=torch.long)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_texts, lengths, labels


def create_dataloaders(data_dir, batch_size=32, max_size=20000, min_freq=2):
    data_dir = Path(data_dir)

    print("Loading training reviews from disk...")
    train_texts, train_labels = load_reviews(data_dir / "train")

    print("Loading test reviews from disk...")
    test_texts, test_labels = load_reviews(data_dir / "test")

    print("Building vocabulary...")
    vocab = Vocabulary()
    vocab.build_vocab(train_texts, max_size=max_size, min_freq=min_freq)

    train_dataset = IMDBDataset(train_texts, train_labels, vocab)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate
    )

    return train_loader, test_loader, vocab
