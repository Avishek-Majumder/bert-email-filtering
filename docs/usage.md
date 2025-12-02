# Usage Guide

This document explains how to use our implementation of:

> **“Harnessing BERT for Advanced Email Filtering in Cybersecurity”**  
> IEEE Xplore: https://ieeexplore.ieee.org/abstract/document/11058531

We focus on **practical commands** and typical workflows: setting up, running baselines, fine-tuning BERT, and inspecting results.

---

## 1. Prerequisites

- Python **3.9+**
- Git
- (Recommended) A GPU with CUDA support for deep learning and BERT
- Installed dependencies:

```bash
pip install -r requirements.txt
```
## 2. Initial Setup

 1. Clone the repository and enter it:
 ```bash
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_NAME>
 ```
 2. Create a virtual environment (optional but recommended):
 ```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
 ```
 3. Install dependencies:
 ```bash
pip install -r requirements.txt
 ```
 4. Configure dataset paths in config/data.yaml:
 ```bash
paths:
  raw_data_path: "data/raw/SMSSpamCollection.csv"

columns:
  text: "text"     # or "v2" depending on your CSV
  label: "label"   # or "v1"
 ```
Place your dataset in data/raw/ (e.g., data/raw/SMSSpamCollection.csv).

## 3. Running Experiments

All commands are assumed to be run from the project root.

3.1 Run all experiments (ML + DL + BERT)

The easiest way to reproduce the full pipeline is:
```bash
python -m scripts.run_all_experiments
# or
python scripts/run_all_experiments.py
```
This will:
 1. Train and evaluate classical ML baselines on TF–IDF features.
 2. Train and evaluate CNN, LSTM, BiLSTM, RNN sequence models.
 3. Fine-tune the BERT classifier.
 4. Aggregate all metrics into a single table (all_results.csv).
You can monitor progress via logs in experiments/logs/.

## 4. Running Individual Components

Sometimes you may only want a subset of experiments. We provide one script per family.
4.1 Classical ML baselines
```
python -m scripts.run_ml_baselines
# or
python scripts/run_ml_baselines.py
```
This calls src.training.train_ml.train_and_evaluate_ml_models and:
 - Loads the dataset and config from config/data.yaml + config/ml.yaml.
 - Builds TF–IDF features.
 - Trains all configured ML models (RF, LR, SVM, XGBoost, GB, NB, KNN).
 - Saves:
  - Per-model metrics JSON: experiments/results/metrics_<model_name>.json
  - Aggregated table: experiments/results/ml_results.csv
  - Trained models: experiments/models/model_<model_name>.joblib
    
4.2 Deep learning baselines (CNN, LSTM, BiLSTM, RNN)
```
python -m scripts.run_dl_baselines
# or
python scripts/run_dl_baselines.py
```
This calls src.training.train_dl.train_and_evaluate_dl_models and:
 - Builds a vocabulary from training texts.
 - Converts texts to integer sequences with padding/truncation.
 - Trains CNN, LSTM, BiLSTM, and RNN models defined in src/models/dl_architectures.py.
 - Saves:
  - Per-architecture metrics JSON: experiments/results/metrics_dl_<arch>.json
  - Aggregated DL results: experiments/results/dl_results.json
  - Model weights: experiments/models/dl_<arch>.pt
  - Vocabulary artifacts: experiments/artifacts/
    
4.3 BERT fine-tuning    
```
python -m scripts.run_bert
# or
python scripts/run_bert.py
```
This calls src.training.train_bert.train_and_evaluate_bert and:
 - Builds BERT-ready datasets (tokenized input_ids, attention_mask, labels).
 - Fine-tunes a pretrained BERT model (default: bert-base-uncased).
 - Evaluates on the test set.
 - Saves:
  - BERT metrics: experiments/results/bert_metrics.json
  - PyTorch state dict: experiments/models/bert_classifier.pt
  - Hugging Face-style model directory: experiments/bert/

## 5. Inspecting Results
5.1 Per-family result files
After running experiments, you can inspect:
- Classical ML:
 - experiments/results/ml_results.csv
- Deep Learning:
 - experiments/results/dl_results.json
- BERT:
 - experiments/results/bert_metrics.json
Each file contains accuracy, precision, recall, and F1-score. Some may also include loss and confusion matrices.

5.2 Aggregated comparison
To aggregate and compare across all model families:
```
python -m src.evaluation.analysis
```
This:
 - Reads the ML / DL / BERT metrics.
 - Writes a combined CSV:
```
experiments/results/all_results.csv
```
The combined table contains:
 - model – model identifier (e.g., random_forest, cnn, bert)
 - category – ml, dl, or bert
 - accuracy, precision, recall, f1
You can open this CSV directly in any spreadsheet tool.

## 6. Evaluation Helpers and Plots

For more detailed evaluation, use:
- src/evaluation/evaluate_models.py – utilities for loading metrics, ranking models, and printing summaries.
- src/evaluation/plots.py – functions to create:
  - F1-score bar charts across models
  - Confusion matrix heatmaps for selected models
Example (Python):
```
from src.evaluation.analysis import aggregate_all_results
from src.evaluation.plots import plot_f1_scores

df = aggregate_all_results()
plot_f1_scores(df, out_path="experiments/results/f1_scores.png")
```

## 7. Customization Workflows
Common custom workflows:

7.1 New dataset (email logs, phishing datasets, etc.)

1. Put the new dataset CSV under data/raw/.
2. Update config/data.yaml:
- paths.raw_data_path
- columns.text, columns.label
- Label mapping under labels.
3. Optionally adjust preprocessing parameters (e.g., max sequence length, stopwords).
4. Re-run the desired scripts (run_ml_baselines.py, run_dl_baselines.py, run_bert.py).

7.2 Hyperparameter experiments
1. Edit config/ml.yaml, config/dl.yaml, or config/bert.yaml.
2. Run the corresponding script(s).
3. Compare metrics in experiments/results/.

7.3 Trying a different transformer
1. In config/bert.yaml, change:
```
model:
  pretrained_model_name: "bert-base-uncased"
```
to a different model (e.g., "roberta-base").
2. Ensure the new model is supported by transformers.
3. Run scripts/run_bert.py again.

## 8. Troubleshooting

- CUDA / GPU issues
  - Set general.device: "cpu" in config/train.yaml to force CPU training.
  - Reduce batch size or max sequence length if you see OOM errors.
- File not found
 - Check config/data.yaml paths.
 - Ensure directories under experiments/ exist; they are created automatically, but .gitignore may hide them until created.

- Slow training
 - For quick tests, reduce:
  - Number of epochs in config/dl.yaml and config/bert.yaml.
  - Dataset size (temporarily sub-sample your data).
If you need more advanced help (e.g., evaluating on custom splits, deploying models, or integrating into a larger system), you can extend the scripts in scripts/ or add new utilities under src/evaluation/ and src/training/.
