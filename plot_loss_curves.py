import matplotlib.pyplot as plt
from train import train_model
from preprocess import prepare_imdb_dataloaders
from imdb_loader import get_imdb_dataset
from utils import set_seed

set_seed(42)
imdb_data = get_imdb_dataset()

# Best model example (BiLSTM, tanh, Adam, 50)
train_loader, test_loader, word2idx = prepare_imdb_dataloaders(imdb_data, seq_len=50, batch_size=32)
model_best, result_best = train_model(train_loader, test_loader,
                                      vocab_size=len(word2idx),
                                      seq_len=50,
                                      arch="bilstm",
                                      activation="tanh",
                                      optimizer_name="adam",
                                      grad_clipping=True,
                                      num_epochs=5)

# Worst model example (RNN, tanh, SGD, 50)
model_worst, result_worst = train_model(train_loader, test_loader,
                                        vocab_size=len(word2idx),
                                        seq_len=50,
                                        arch="rnn",
                                        activation="tanh",
                                        optimizer_name="sgd",
                                        grad_clipping=False,
                                        num_epochs=5)

plt.plot(result_best["train_loss_curve"], label="Best (BiLSTM-Adam)")
plt.plot(result_worst["train_loss_curve"], label="Worst (RNN-SGD)")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch for Best and Worst Models")
plt.legend()
plt.savefig("results/plots/loss_vs_epoch.png")
plt.show()
