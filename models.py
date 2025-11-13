# src/models.py
import torch
import torch.nn as nn


def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation {name}")

class BaseRNNClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim=100,
                 hidden_size=128,
                 num_layers=2,
                 rnn_type="rnn",
                 bidirectional=False,
                 dropout=0.5,
                 activation="tanh"):
        super().__init__()
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.activation = get_activation(activation)

        if rnn_type.lower() == "rnn":
            self.rnn = nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity="tanh",  # internal RNN nonlinearity (PyTorch only tanh/relu here)
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type must be 'rnn' or 'lstm'")

        self.dropout = nn.Dropout(dropout)

        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, 1)
        self.out_act = nn.Sigmoid()  # final binary classification
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    
        self.rnn_type = rnn_type.lower()

    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embed(x)  # [batch, seq_len, embed_dim]

        if self.rnn_type == "lstm":
            output, (h_n, c_n) = self.rnn(emb)
        else:
            output, h_n = self.rnn(emb)

        # h_n shape: [num_layers * num_directions, batch, hidden_size]
        # take last layer's hidden state(s)
        if self.bidirectional:
            # concat last layer forward + backward
            h_last_fwd = h_n[-2,:,:]
            h_last_bwd = h_n[-1,:,:]
            h_cat = torch.cat([h_last_fwd, h_last_bwd], dim=1)
            rep = h_cat
        else:
            rep = h_n[-1,:,:]  # [batch, hidden]

        rep = self.dropout(rep)
        rep = self.activation(rep)
        logits = self.fc(rep).squeeze(1)  # [batch]
        prob = self.out_act(logits)
        return prob
