# Sentiment Analysis: RNN vs. LSTM Benchmark

A comparative study evaluating the performance, convergence rate, and generalization of **Vanilla Recurrent Neural Networks (RNN)** versus **Long Short-Term Memory (LSTM)** networks for binary sentiment classification on the IMDB Large Movie Review Dataset.

---

## Overview

Vanilla RNNs process sequences via an internal hidden state, but suffer from vanishing/exploding gradients on long sequences due to repeated matrix multiplications in backpropagation-through-time (BPTT). LSTMs address this with a memory cell and three gates:

- **Forget Gate**: determines what to discard from the previous cell state.
- **Input Gate**: determines which new values to store in the cell state.
- **Output Gate**: controls what part of the updated cell state is emitted as the hidden state.

This repository benchmarks both architectures under an identical experimental setup (dataset, vocabulary, embedding size, hidden dimension, layer depth, learning rate, optimizer).

---

## Architecture and Preprocessing

- **Dataset**: Large Movie Review Dataset (`aclImdb`) — 25,000 training and 25,000 test reviews labeled positive (1) / negative (0).
- **Preprocessing**: strip HTML tags, remove non-alphanumerics, lowercase, trim whitespace; frequency-filtered vocabulary with `<pad>` (0) and `<unk>` (1) tokens; dynamic padding and `pack_padded_sequence` to ignore padded tokens.
- **Model Pipeline**:
  - `nn.Embedding(vocab_size=20,002, embedding_dim=128, padding_idx=0)`
  - 2-layer bidirectional RNN or LSTM (`hidden_dim=128`, `dropout=0.3`), forward/backward hidden states concatenated (dim 256)
  - Classification head: linear layer (`256 -> 2`) with `nn.CrossEntropyLoss`

---

## Experimental Results

Both models were trained on CUDA for 5 epochs using the Adam optimizer (`lr=1e-3`) with gradient clipping (`clip=5.0`) and batch size 32.

### Final Performance Summary

| Metric | Vanilla RNN | Bidirectional LSTM | Delta |
| :--- | :---: | :---: | :---: |
| **Trainable Parameters** | 2,725,634 | 3,220,226 | +494,592 (+18.1%) |
| **Final Train Loss** | 0.4234 | **0.1605** | -0.2629 |
| **Final Train Accuracy** | 80.95% | **94.03%** | +13.08% |
| **Final Validation Loss** | 0.5721 | **0.3479** | -0.2242 |
| **Final Validation Accuracy** | 72.29% | **87.02%** | **+14.73%** |
| **Best Validation Accuracy** | 75.25% | **88.11%** | **+12.86%** |

### Training Curves

![RNN vs LSTM Loss and Accuracy Comparison](rnn_vs_lstm_comparison.png)

### Key Findings

1. **Long-Term Dependencies**: RNN hidden states decay over long reviews, capping validation accuracy at **75.25%**, while the LSTM cell state preserves long-range sentiment, reaching **88.11%**.
2. **Convergence**: The LSTM fits the training data much better, reaching loss **0.1605** (94.03% accuracy) versus **0.4234** (80.95%) for the RNN.
3. **Parameter Efficiency**: Despite ~4x larger recurrent weights (4 internal gates), the shared embedding matrix dominates both networks, so the LSTM gains **~14.7%** validation accuracy for only an **18.1%** parameter increase.

---

## Project Structure

```text
├── data.py                   # Preprocessing, vocabulary building, dataset loading, collate function
├── model.py                  # SentimentClassifier (RNN or LSTM)
├── trainer.py                # Training, validation, logging, checkpointing
├── main.py                   # CLI entry point: benchmarking and plotting
├── rnn_vs_lstm_comparison.png# Generated comparison plot
├── aclImdb/                  # IMDB dataset (gitignored)
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

---

## Getting Started

1. **Install dependencies** (Python 3.8+, PyTorch >= 2.0):
   ```bash
   pip install torch tqdm matplotlib numpy
   ```

2. **Download the dataset**: the [`aclImdb` (Stanford Large Movie Review) dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) is not included in the repository (50,000 text files, gitignored). Download the archive above and extract it to the project root:

   The `aclImdb/` folder must contain `train/` and `test/` subdirectories, each with `pos/` and `neg/` folders.

3. **Run the benchmark** (trains RNN and LSTM, prints the comparison table, and generates the plot):
   ```bash
   python main.py
   ```

4. **Optional arguments** (defaults shown):
   ```bash
   python main.py --data_dir ./aclImdb --epochs 5 --batch_size 32 \
     --hidden_dim 128 --num_layers 2 --dropout 0.3 --lr 0.001 \
     --seed 42 --plot
   ```
