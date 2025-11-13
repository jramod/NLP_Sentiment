import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/metrics.csv")

# Plot 1: Accuracy & F1 vs Sequence Length (only for LSTM + Adam + GradClip=True)
subset = df[(df["Model"]=="lstm") & (df["Optimizer"]=="adam") & (df["GradClipping"]==True)]
plt.plot(subset["SeqLen"], subset["Accuracy"], marker='o', label="Accuracy")
plt.plot(subset["SeqLen"], subset["F1"], marker='o', label="F1-score")
plt.xlabel("Sequence Length")
plt.ylabel("Score")
plt.title("LSTM (Adam, GradClip=True): Accuracy & F1 vs Sequence Length")
plt.legend()
plt.savefig("results/plots/accuracy_f1_vs_seq.png")
plt.show()
