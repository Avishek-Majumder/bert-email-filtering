# Experimental and Implementation Assumptions

This document captures the main assumptions underlying our implementation of:

> **“Harnessing BERT for Advanced Email Filtering in Cybersecurity”**  
> IEEE Xplore: https://ieeexplore.ieee.org/abstract/document/11058531

We keep these assumptions explicit so that users can understand and, if needed, relax or modify them.

---

## 1. Problem Definition

1. We treat the task as a **binary text classification** problem:
   - Class 0 → *legitimate* (ham / non-spam)
   - Class 1 → *malicious* (spam / unwanted content)

2. The implementation focuses on **message-level classification**:
   - Each row in the dataset is a single message (SMS or email-like text).
   - No thread-level, user-level, or conversation-level modeling is done.

3. The primary goal is to **compare models** (ML / DL / BERT) under consistent conditions, not to design a production system (e.g., with deployment, latency constraints, or live feedback loops).

---

## 2. Dataset & Labels

1. By default, we use the **SMS Spam Collection** dataset as a **proxy** for email spam/phishing detection:
   - The dataset should be provided as a CSV and configured via `config/data.yaml`.
   - Column names for text and labels are defined in `config/data.yaml` (e.g., `text`, `label`).

2. Label mapping is assumed to be **binary** and configured as:

   ```yaml
   labels:
     ham: 0
     spam: 1
   ```
If your dataset uses different label strings (e.g., normal, phishing), you should update this mapping.

3. We assume that:

 - There are no missing labels.

 - Text fields are non-empty strings (empty or NaN rows should be cleaned before training or handled in preprocessing).

4. The default train/test split is fixed at 70/30, stratified on label_id to preserve class balance.

## 3. Preprocessing & Feature Engineering
3.1 General Text Assumptions

1. Messages are in English (or predominantly English).
Non-English texts are not explicitly handled with language-specific pipelines.
2. Basic preprocessing is controlled by config/data.yaml and may include:
 - Lowercasing
 - URL / email / number masking or removal
 - Removal of repeated whitespace
 - Optional stopword removal for classical ML / DL

3. For classical ML and DL baselines, we assume that:
 - Simple normalization is sufficient (no heavy domain-specific NLP).
 - Tokenization is whitespace/token-based, as implemented in our preprocessing utilities.
   
3.2 TF–IDF Features (ML models)
1. TF–IDF is fit only on the training texts and then applied to test texts.
2. The fitted TF–IDF vectorizer is persisted under experiments/artifacts/ and reused for evaluation.
3. We assume that the default hyperparameters (ngram range, max_features, etc.) defined in config/data.yaml are adequate for the baseline comparisons.

3.3 Sequence Features (DL models)
1. Vocabulary is built only from training texts, with vocab size and sequence length defined in config/data.yaml.
2. We assume that:
 - Token IDs 0, 1, etc. are reserved for padding/unknown tokens as defined by Vocabulary.
 - Sequence truncation/padding is acceptable when messages exceed the maximum length.

## 4. Model & Training Assumptions
4.1 Classical ML Models

1. We implement the following models using scikit-learn / XGBoost:
 - Random Forest
 - Logistic Regression
 - SVM
 - XGBoost
 - Gradient Boosting
 - Naive Bayes (Multinomial/Bernoulli)
 - KNN
2. Hyperparameters are entirely config-driven (config/ml.yaml) and not auto-tuned in this codebase.
3. For some models (Logistic Regression, SVM, KNN), we optionally apply feature scaling (StandardScaler) depending on use_feature_scaling in config/ml.yaml.

4.2 Deep Learning Sequence Models (CNN / LSTM / BiLSTM / RNN)

1. All DL models are implemented in PyTorch and share:
 - A trainable embedding layer.
 - Cross-entropy loss.
 - Optimization settings defined in config/dl.yaml.
2. We assume:
 - Batch training, with a fixed number of epochs.
 - Single-task, single-label classification.
3. Regularization:
 - Dropout values and other regularization parameters are set in config/dl.yaml.
 - No advanced regularization (e.g., MixUp, label smoothing) is used by default.

4.3 BERT-based Classifier
1. We use a Hugging Face Transformers model for sequence classification:
 - Default backbone: bert-base-uncased (can be changed in config/bert.yaml).
 - Classification head with num_labels = 2.
2. Training assumptions:
 - Fine-tuning for a small number of epochs (default: 3).
 - Batch size and learning rate consistent with common BERT fine-tuning practice.
 - Optional fp16 mixed precision on GPU if enabled in config/bert.yaml.
3. We assume a moderate GPU is available for BERT training; training on CPU is possible but slow.

## 5. Reproducibility & Randomness

1. We set a global random seed (Python, NumPy, PyTorch) via seed_everything in src/utils/training_utils.py.
2. Deterministic behavior is requested from cuDNN when possible, but:
 - Some operations may still introduce minor non-determinism depending on hardware / CUDA / PyTorch versions.
3. Re-running experiments with the same configs and seed should yield similar (but not necessarily bit-identical) metrics.

## 6. Environment & Resource Assumptions
   
1. Python version: 3.9+ is recommended.
2. The environment has:
 - Sufficient RAM to hold the dataset and TF–IDF features in memory.
 - Disk space for logs, models, vectorizers, and aggregated results under experiments/.
3. For BERT:
 - A single GPU with at least ~6–8 GB VRAM is recommended for the default settings.
 - If running on CPU, users may need to reduce batch size or sequence length.

## 7. Evaluation & Metrics

1. We evaluate using:
 - Accuracy
 - Precision
 - Recall
 - F1-score
Confusion matrix
2. For binary classification, we treat spam as the positive class (label 1) in our metrics.
3. Aggregated results across ML / DL / BERT are compared using F1 as the primary ranking metric, consistent with the emphasis on balanced performance.

## 8. Limitations of These Assumptions

1. The SMS dataset is a proxy for real-world email spam; domain shift is expected in production email environments.
2. The implementation does not include:
 - Multi-lingual or code-mixed handling.
 - Online / streaming learning, concept drift handling, or adversarial robustness.
3. Hyperparameters are designed to be reasonable defaults, not the result of exhaustive search.
   
Users who want to deploy in production or on different datasets are encouraged to:
 - Revisit these assumptions.
 - Adapt configs and models.
 - Add domain-specific preprocessing and evaluation as needed.    
