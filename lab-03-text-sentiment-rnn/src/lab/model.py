import torch
import torch.nn as nn


class SentimentRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        num_layers,
        dropout,
        num_classes=2,
        bidirectional=False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, (h_n, _) = self.lstm(x)

        if self.lstm.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]

        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits