# NLP_Sentiment
Sentiment Classification is a core NLP task that involves categorizing the emotional tone of a piece of text such as a movie review into classes like positive/ negative. In this project, I will implement and evaluate multiple Recurrent Neural Network (RNN) architectures for sentiment classification, treating it as a sequence classification problem.
# Comparative Analysis of RNN Architectures for Sentiment Classification

This project implements and evaluates multiple Recurrent Neural Network (RNN) architectures — **Vanilla RNN**, **LSTM**, and **Bidirectional LSTM (BiLSTM)** — for sentiment classification on the IMDb Movie Review Dataset.  
The goal is to compare model performance (Accuracy, F1, training time) across variations in sequence length, activation function, and optimization strategy.

---

## 🧩 Setup Instructions

### Python Version
- **Python 3.9 or 3.10**
- Recommended environment: conda (miniforge / anaconda)

### Dependencies
Install all dependencies using pip or conda:

```bash
pip install torch numpy scikit-learn matplotlib psutil
pip install torchvision torchaudio

python experiment.py

