# Harnessing BERT for Advanced Email Filtering in Cybersecurity – Reproducible Implementation

In this repository, we implement the complete experimental pipeline for our work:

> **“Harnessing BERT for Advanced Email Filtering in Cybersecurity”**  
> IEEE Xplore: https://ieeexplore.ieee.org/abstract/document/11058531

We provide a unified, reproducible codebase that covers:

- Classical ML baselines with TF–IDF features  
- Deep sequence models (CNN, LSTM, BiLSTM, RNN)  
- A BERT-based classifier fine-tuned for spam/ham detection  

Although the original context is email filtering in cybersecurity, we use the standard **SMS Spam Collection** dataset as a benchmark for text spam detection and model comparison.

---

## 1. Repository Structure

```text
.
├── config/
│   ├── data.yaml          # Dataset paths and preprocessing settings
│   ├── train.yaml         # Global training, logging, and paths
│   ├── ml.yaml            # Classical ML model hyperparameters
│   ├── dl.yaml            # CNN/LSTM/BiLSTM/RNN hyperparameters
│   └── bert.yaml          # BERT model and training configuration
├── data/
│   └── raw/               # Place the SMS Spam Collection dataset here
├── experiments/
│   ├── logs/              # Training logs
│   ├── models/            # Saved model weights / artifacts
│   └── results/           # Metrics and aggregated result tables
├── scripts/
│   └── run_all_experiments.py  # Orchestrates full ML + DL + BERT pipeline
├── src/
│   ├── data/
│   │   ├── datasets.py        # Dataset loading and label mapping
│   │   ├── split.py           # Train/test splitting (stratified)
│   │   ├── sequence_dataset.py# Vocabulary + sequence Dataset for DL models
│   │   └── bert_dataset.py    # BERT-ready Dataset (input_ids, attention_mask, labels)
│   ├── evaluation/
│   │   ├── metrics.py         # Accuracy / Precision / Recall / F1 / confusion matrix
│   │   └── analysis.py        # Aggregation of ML, DL, BERT results
│   ├── features/
│   │   ├── preprocessing.py   # Text preprocessing for classical + DL + BERT
│   │   └── tfidf_vectorizer.py# TF–IDF fitting, saving, and transforming
│   ├── models/
│   │   ├── ml_models.py       # RF, LR, SVM, XGBoost, GB, NB, KNN factories
│   │   ├── dl_architectures.py# CNN, LSTM, BiLSTM, RNN architectures
│   │   └── bert_classifier.py # BERT sequence classifier factory
│   ├── training/
│   │   ├── train_ml.py        # Classical ML training + evaluation
│   │   ├── train_dl.py        # DL models training + evaluation
│   │   └── train_bert.py      # BERT fine-tuning + evaluation
│   └── utils/
│       └── training_utils.py  # Config loading, seeding, logging, device selection
├── requirements.txt
└── README.md
```
2. Installation

We recommend Python 3.9+.
```bash
# Clone your repository
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_NAME>

# (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install required packages
pip install -r requirements.txt
```
The requirements.txt includes all dependencies for:
. NumPy / Pandas (data handling)
. scikit-learn + XGBoost (classical ML)
. PyTorch (DL models)
. Hugging Face Transformers (BERT)
. NLTK (for optional preprocessing steps)

3. Dataset Preparation

By default, we use the SMS Spam Collection dataset. You can download it from the UCI repository, then point config/data.yaml to the correct CSV file.
1. Download the dataset and place it under data/raw/, for example:
```text
data/raw/SMSSpamCollection.csv
```
2. Open config/data.yaml and adjust the paths if needed:
```bash
paths:
  raw_data_path: "data/raw/SMSSpamCollection.csv"
  # ...
columns:
  text: "text"        # or "v2" depending on your CSV format
  label: "label"      # or "v1"
```
3. Ensure the label mapping in config/data.yaml matches your file:
```yaml
labels:
  ham: 0
  spam: 1
```
If you use a different spam/email dataset, you can still reuse the full pipeline by updating the paths and column names here.

4. Configuration Overview
All experiments are driven by YAML configs:
```yaml
config/data.yaml
```
. File paths, column names, label mapping
. Preprocessing options (lowercasing, URL removal, stopwords, sequence length, vocab limits)
```yaml
config/train.yaml
```
. Global random seed, determinism, number of workers
. Device preference (cuda / cpu)
. Logging settings (console + file)
. Output paths: experiments/logs, experiments/results, experiments/models
. Saving behaviour (whether to overwrite existing models)
```yaml
config/ml.yaml
```
. Hyperparameters for RF, LR, SVM, XGBoost, Gradient Boosting, Naive Bayes, KNN
. Flags for class balancing (class_weight=balanced) and feature scaling
```yaml
config/dl.yaml
```
. Embedding dimension, sequence length, vocab size
. Architecture-specific settings for CNN, LSTM, BiLSTM, RNN
. Optimization parameters (optimizer, learning rate, schedulers, epochs)
```yaml
config/bert.yaml
```
. Pretrained backbone (e.g., bert-base-uncased)
. Max sequence length, dropout, number of labels
. Fine-tuning settings (batch size, learning rate, warmup ratio, epochs, fp16, best-model tracking)
We treat these configs as the single source of truth, so we can reproduce and tweak our experiments without touching code.

5. Running Experiments
All commands below are assumed to be executed from the project root.

5.1 Run all experiments (ML + DL + BERT)
This orchestrates the entire pipeline end-to-end:
```bash
python -m scripts.run_all_experiments
# or
python scripts/run_all_experiments.py
```
This will:
1. Train and evaluate all classical ML models on TF–IDF features.
2. Train and evaluate CNN, LSTM, BiLSTM, and RNN sequence models.
3. Fine-tune the BERT classifier.
4. Aggregate metrics into a single results table.
   
5.2 Classical ML only
```bash
python -m src.training.train_ml
# or
python src/training/train_ml.py
```
Outputs:
. Per-model metrics: experiments/results/metrics_<model_name>.json
. Aggregated ML table: experiments/results/ml_results.csv
. Trained models: experiments/models/model_<model_name>.joblib

5.3 Deep Learning (CNN/LSTM/BiLSTM/RNN) only
```bash
python -m src.training.train_dl
# or
python src/training/train_dl.py
```
Outputs:
. Per-architecture metrics: experiments/results/metrics_dl_<arch>.json
. Aggregated DL results: experiments/results/dl_results.json
. Model weights: experiments/models/dl_<arch>.pt
. Saved vocabulary: experiments/artifacts/vocab.json (via save_vocab)

5.4 BERT fine-tuning only
```bash
python -m src.training.train_bert
# or
python src/training/train_bert.py
```
Outputs:
. BERT metrics: experiments/results/bert_metrics.json
. PyTorch weights: experiments/models/bert_classifier.pt
. Hugging Face format model: experiments/bert/ (config + weights, ready for from_pretrained)

6. Result Aggregation and Analysis
After running experiments, we aggregate all results into a single table:
```bash
python -m src.evaluation.analysis
```

This:
. Reads:
 . ml_results.csv (classical ML)
 . dl_results.json (CNN/LSTM/BiLSTM/RNN)
 . bert_metrics.json (BERT)

. Produces a combined CSV:
```text
experiments/results/all_results.csv
```
The combined table contains:
. model – model identifier (e.g., random_forest, cnn, bert)
. category– ml, dl, or bert
. accuracy, precision, recall, f1
. Optionally additional fields such as test_loss or confusion matrices
We can directly use this file to reconstruct the comparative performance tables from our paper.

7. Extending and Customizing
A few typical modifications:

. Different dataset
```text
Update config/data.yaml paths, column names, and labels.
```
. Hyperparameter tuning
```text
Adjust config/ml.yaml, config/dl.yaml, or config/bert.yaml and rerun the corresponding training script.
```
. New architecture
```text
Add a new model in src/models/dl_architectures.py and register it in build_dl_model, then invoke it from train_dl.py.
```
. Alternative transformer
```text
Change pretrained_model_name in config/bert.yaml to another Hugging Face model (e.g., roberta-base), keeping other settings compatible.
```
8. Paper and Citation
This repository is the official implementation of our experiments from:
```text
“Harnessing BERT for Advanced Email Filtering in Cybersecurity”
IEEE Xplore: https://ieeexplore.ieee.org/abstract/document/11058531
```
If you build on this implementation in academic work, please cite the paper using the official IEEE citation format from the IEEE Xplore page above.
You may also reference this repository as the implementation of the experimental pipeline used in the paper.

9. Contact / Issues
If you encounter issues, have questions about the experiments, or want to adapt the pipeline to a different dataset or setting, feel free to:
 . Open an issue in the repository, or
 . Fork and customize the configuration files and training scripts as needed.
We designed the codebase to be modular and config-driven so that extending or reusing the components is straightforward.
