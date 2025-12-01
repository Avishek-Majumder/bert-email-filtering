# Harnessing BERT for Advanced Email Filtering in Cybersecurity – Reproducible Implementation

In this repository, we implement and reproduce the experiments from the paper:

> **“Harnessing BERT for Advanced Email Filtering in Cybersecurity”**  
> Avishek Majumder, Tanjim Mahmud, Tikle Barua, Rishita R. Kundu, Mohammad Shahadat Hossain, Md. Minarul Islam, Karl Andersson  
> (ICICT 2024)

Our goal is to provide a clean, fully reproducible codebase that covers all models and experiments described in the paper, including:

- **Classical ML models**: Random Forest (RF), Logistic Regression (LR), Support Vector Machine (SVM), XGBoost (XGB), Gradient Boosting (GB), Naive Bayes (NB), and K-Nearest Neighbors (KNN).
- **Deep learning models**: CNN, LSTM, BiLSTM, and RNN.
- **Transformer-based model**: BERT fine-tuning for binary spam/ham email (or SMS) classification.

We work with the well-known **SMS Spam Collection** dataset and follow the same preprocessing steps, train/validation split, and evaluation metrics as reported in the paper.

---

## 1. Project Overview

We focus on binary spam detection in the context of cybersecurity, comparing a broad spectrum of machine learning and deep learning models against a fine-tuned BERT baseline.

Key aspects:

- **Dataset**: SMS Spam Collection (4,827 ham and 747 spam samples).
- **Preprocessing**: text cleaning, tokenization, stopword removal, and stemming/lemmatization (for non-BERT models).
- **Features**:
  - TF–IDF vectors for classical ML models.
  - Token sequences + embeddings for CNN/LSTM/BiLSTM/RNN.
  - BERT tokenizer and embeddings for the transformer model.
- **Metrics**: Accuracy, Precision, Recall, F1-score, and confusion matrix analysis.

Our implementation is structured to make it easy to rerun all experiments and compare results across models.

---

## 2. Repository Structure

At a high level, the repository is organized as follows:

```text
config/           # YAML configuration files (data, ML, DL, BERT, training)
data/
  raw/            # Original SMS Spam Collection dataset CSV
  processed/      # Preprocessed artifacts (TF-IDF, vocab, etc.)
src/
  data/           # Dataset loading, splitting, PyTorch datasets
  features/       # Text preprocessing and feature extraction (TF-IDF)
  models/         # ML, DL, and BERT model definitions
  training/       # Training scripts for ML, DL, and BERT
  evaluation/     # Metrics, evaluation utilities, and plotting
  utils/          # Shared utilities (seeding, logging, paths)
scripts/          # Command-line entry points to run experiments
experiments/      # Logs, saved models, results, and plots
docs/             # Assumptions, extended usage docs
notebooks/        # Optional exploratory notebooks
tests/            # Optional unit tests
```
We will gradually fill in each component as we progress through the steps.

3. Getting Started
3.1. Environment

We recommend:
```text
Python: 3.10 or later

Core libraries (to be listed in requirements.txt):

pandas, numpy, scikit-learn

torch, torchvision, torchaudio

transformers

matplotlib, seaborn (optional, for plots)

pyyaml, tqdm
```
Once requirements.txt is available, you will be able to install dependencies via:
```bash
pip install -r requirements.txt
```
3.2. Dataset
Download the SMS Spam Collection dataset (Kaggle or UCI repository) as a CSV file and place it under:
```bash
data/raw/sms_spam_collection.csv
```
We will make the exact path and column names configurable in config/data.yaml.

4. Experiments

We design this repository so that each group of models can be run via a dedicated script:
```bash
scripts/run_ml_baselines.py – trains and evaluates all classical ML models on TF–IDF features.

scripts/run_dl_baselines.py – trains and evaluates CNN/LSTM/BiLSTM/RNN models.

scripts/run_bert.py – fine-tunes BERT on the dataset and evaluates performance.

scripts/run_all_experiments.py – (optional) runs all experiments and aggregates metrics.
```
Each script will:
```text
1. Load configuration from config/.
2. Prepare data and features.
3. Train the model(s).
4. Evaluate using accuracy, precision, recall, and F1-score.
5. Save metrics, plots, and (optionally) model checkpoints under experiments/.
```
Exact commands and examples will be added once all components are implemented.

5. Reproducibility

We will:
```text
1. Use a stratified 70/30 train–test split as in the paper.
2. Fix random seeds for NumPy, PyTorch, and Python’s random module.
3. Save all configuration files (YAML) under version control.
4. Export metrics and plots so users can compare their results with those reported in the paper.
```
6. Citation

If you use this implementation or build upon our work, please cite the paper:
```bibtex
@inproceedings{majumber2024harnessingbert,
  title     = {Harnessing BERT for Advanced Email Filtering in Cybersecurity},
  author    = {Majumder, Avishek and Mahmud, Tanjim and Barua, Tikle and Kundu, Rishita R. and Hossain, Mohammad Shahadat and Islam, Md. Minarul and Andersson, Karl},
  booktitle = {Proceedings of the International Conference on Information and Communication Technology (ICICT)},
  year      = {2024}
}
```
(We will refine the BibTeX entry with the exact conference details once all metadata is finalized.)

7. Roadmap

We will implement the repository in the following stages:
```text
1. Repository structure and configuration files.
2. Data loading and preprocessing.
3. Feature extraction (TF–IDF, token sequences, BERT inputs).
4. ML, DL, and BERT model implementations.
5. Training and evaluation pipelines.
6. Visualization and result analysis.
7. Final documentation and polish.
```
Each stage will correspond to clearly defined steps and sub-steps so that the implementation remains transparent and easy to follow.
