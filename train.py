# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import set_seed, epoch_timer
from evaluate import evaluate_model
from models import BaseRNNClassifier

def get_optimizer(name, model, lr=1e-3):
    name = name.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    raise ValueError(f"Unknown optimizer {name}")

def train_model(
        train_loader,
        test_loader,
        vocab_size,
        seq_len,
        arch="rnn",             # "rnn", "lstm", "bilstm"
        activation="tanh",      # "relu", "tanh", "sigmoid"
        optimizer_name="adam",  # "adam","sgd","rmsprop"
        grad_clipping=False,
        clip_value=1.0,
        num_epochs=5,
        batch_size=32,
        embed_dim=100,
        hidden_size=64,
        num_layers=2,
        dropout=0.5,
        seed=42,
        device=None
    ):
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bidirectional = (arch.lower() == "bilstm")
    rnn_type = "lstm" if arch.lower() in ["lstm","bilstm"] else "rnn"

    model = BaseRNNClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        dropout=dropout,
        activation=activation
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = get_optimizer(optimizer_name, model)

    epoch_times = []
    train_losses_per_epoch = []

    for epoch in range(num_epochs):
        model.train()
        start_t = time.time()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            probs = model(xb)
            loss = criterion(probs, yb)
            loss.backward()

            if grad_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        end_t = time.time()

        epoch_time_sec = epoch_timer(start_t, end_t)
        epoch_times.append(epoch_time_sec)

        avg_loss = running_loss / len(train_loader.dataset)
        train_losses_per_epoch.append(avg_loss)

        # You can print live training logs if you want
        # print(f"Epoch {epoch+1}/{num_epochs} Loss={avg_loss:.4f} Time={epoch_time_sec:.2f}s")

    # Final eval after training
    acc, f1 = evaluate_model(model, test_loader, device=device)

    # aggregate stats
    avg_epoch_time = sum(epoch_times)/len(epoch_times)

    results = {
        "arch": arch,
        "activation": activation,
        "optimizer": optimizer_name,
        "seq_len": seq_len,
        "grad_clipping": grad_clipping,
        "accuracy": acc,
        "f1": f1,
        "epoch_time_sec": avg_epoch_time,
        "train_loss_curve": train_losses_per_epoch
    }
    return model, results
