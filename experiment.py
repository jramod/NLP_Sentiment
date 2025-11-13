import os, csv
from utils import get_hardware_report, set_seed
from preprocess import prepare_imdb_dataloaders
from train import train_model
from imdb_loader import get_imdb_dataset  # <-- NEW

set_seed(42)

print("Loading IMDb (this may download once)...")
imdb_data = get_imdb_dataset()
print("Loaded IMDb. Train size:", len(imdb_data["train_texts"]), "Test size:", len(imdb_data["test_texts"]))

configs = [
    {"arch":"rnn","activation":"tanh","optimizer_name":"adam","seq_len":50,"grad_clipping":True},
    {"arch":"lstm","activation":"tanh","optimizer_name":"adam","seq_len":50,"grad_clipping":True},
    {"arch":"bilstm","activation":"tanh","optimizer_name":"adam","seq_len":50,"grad_clipping":True},
    {"arch":"lstm","activation":"relu","optimizer_name":"adam","seq_len":50,"grad_clipping":True},
    {"arch":"lstm","activation":"sigmoid","optimizer_name":"adam","seq_len":50,"grad_clipping":True},
    {"arch":"lstm","activation":"tanh","optimizer_name":"sgd","seq_len":50,"grad_clipping":True},
    {"arch":"lstm","activation":"tanh","optimizer_name":"rmsprop","seq_len":50,"grad_clipping":True},
    {"arch":"lstm","activation":"tanh","optimizer_name":"adam","seq_len":25,"grad_clipping":True},
    {"arch":"lstm","activation":"tanh","optimizer_name":"adam","seq_len":100,"grad_clipping":True},
    {"arch":"lstm","activation":"tanh","optimizer_name":"adam","seq_len":50,"grad_clipping":False}
]

configs = [
    {"arch":"lstm","activation":"tanh","optimizer_name":"adam","seq_len":100,"grad_clipping":True,"hidden_size":128,"num_epochs":6},
    {"arch":"bilstm","activation":"tanh","optimizer_name":"adam","seq_len":200,"grad_clipping":True,"hidden_size":128,"num_epochs":8},
    {"arch":"lstm","activation":"relu","optimizer_name":"adam","seq_len":200,"grad_clipping":True,"hidden_size":128,"num_epochs":6},
]


os.makedirs("results", exist_ok=True)
metrics_path = "results/metrics.csv"

with open(metrics_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model","Activation","Optimizer","SeqLen","GradClipping","Accuracy","F1","EpochTime"])

for cfg in configs:
    seq_len = cfg["seq_len"]
    print(f"\n=== Running {cfg} ===")

    # build dataloaders for this seq_len using our preprocessing pipeline
    train_loader, test_loader, word2idx = prepare_imdb_dataloaders(
        imdb_data,
        seq_len=seq_len,
        batch_size=32,
        vocab_size=10000,
        seed=42
    )

    model, result = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        vocab_size=len(word2idx),
        seq_len=seq_len,
        arch=cfg["arch"],
        activation=cfg["activation"],
        optimizer_name=cfg["optimizer_name"],
        grad_clipping=cfg["grad_clipping"],
        num_epochs=8,         # start with 1 epoch just to confirm pipeline works
        batch_size=32,
        embed_dim=100,
        hidden_size=64,
        num_layers=2,
        dropout=0.5,
        seed=42
    )

    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            cfg["arch"], cfg["activation"], cfg["optimizer_name"],
            seq_len, cfg["grad_clipping"],
            f"{result['accuracy']:.4f}",
            f"{result['f1']:.4f}",
            f"{result['epoch_time_sec']:.2f}"
        ])

print("\n✅ All experiments complete!")
print(f"Results saved to {metrics_path}")
print(get_hardware_report())
