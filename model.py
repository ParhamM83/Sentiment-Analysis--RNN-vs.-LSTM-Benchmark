import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        padding_idx=0,
        cell_type="rnn",
        bidirectional=True
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.bidirectional = bidirectional

        if self.cell_type not in ("rnn", "lstm"):
            raise ValueError(f"Unsupported cell_type '{cell_type}'. Choose 'rnn' or 'lstm'.")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        if self.cell_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.cell_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), 2)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        if self.cell_type == "lstm":
            _, (hidden, _) = self.recurrent(packed)
        else:
            _, hidden = self.recurrent(packed)

        if self.bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            final_hidden = hidden[-1]

        final_hidden = self.dropout(final_hidden)
        logits = self.classifier(final_hidden)

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alias for compatibility with notebook guide
SentimentRNN = SentimentClassifier