# modernbertdos
ModerBERT DoS tool

## ModernBERT-DDOS Classification Tool

This repository contains a tool for fine-tuning a transformer-based model, **ModernBERT**, to classify network traffic into different types based on labeled data. The model is trained on a labeled dataset (CIC-DDoS2019 and one specifically crafted for SSL) for detecting various types of network traffic such as DDoS, normal traffic (BENIGN), and others (NetBIOS, LDAP, Syn, MSSQL, UDP, SSL).

## Evaluation

To evaluate the model's performance on new data, follow the steps below using the provided code.

### Requirements
- Python 3.x
- The following libraries:
  - `torch`
  - `numpy`
  - `pandas`
  - `transformers`
  - `datasets`
  - `scikit-learn`

### Usage
Run the following code to evaluate the fine-tuned ModernBERT model:

```python modernbertdos.py <dataset>```

To run the test case, use the following command:

```python modernbertdos.py testdataset.csv```

## Overview

The tool leverages **Hugging Face's Transformers** library and fine-tunes **ModernBERT** for multi-class classification tasks. The goal is to classify network traffic types from a CSV dataset that includes different features of network packets.

The tool can:
- Use **ModernBERT** to classify the traffic types into one of the predefined categories.

# Training

## Features

- **Model Fine-Tuning**: Uses **ModernBERT** pre-trained on text data and fine-tunes it for network traffic classification.
- **Early Stopping**: Monitors the performance of the model on the validation set to stop training early if improvements are minimal.
- **Model Metrics**: Tracks **Accuracy**, **F1 Score**, **Precision**, and **Recall** during training.
- **Model Saving and Upload**: Saves the fine-tuned model and tokenizer, and uploads them to **Hugging Face Hub** for easy access. [ModernBERT-DDOS Model on Hugging Face](https://huggingface.co/ccaug/modernbert_ddos)

## Requirements

To run the notebook and train the model, you need the following libraries:

- `transformers`
- `datasets`
- `scikit-learn`
- `torch`
- `google-colab` (for Google Colab environments)

You can install them with:

pip install datasets scikit-learn transformers

Setup
1. Use moderbertdos_training.ipynb to train the model.

2. Open the Jupyter Notebook for training the model.

3. Upload the dataset file fulldatasetsmall.csv when prompted in the notebook.

4. Follow the steps in the notebook to start the training process.

Dataset
- Input: The dataset should be a CSV file (fulldatasetsmall.csv) containing network traffic features and labels.
- Label Categories: The dataset contains traffic types labeled as:
  - NetBIOS
  - BENIGN
  - LDAP
  - Portmap
  - Syn
  - MSSQL
  - UDP
  - SSL

The model uses these labels to classify network traffic.

Training Process
1. Data Preparation:
   - The dataset is first cleaned, and labels are mapped to numerical values.
   - The data is then split into training, validation, and testing datasets.

2. Model Setup:
   - The ModernBERT model is used for multi-class classification. It is fine-tuned on the network traffic dataset.

3. Training:
   - The training process uses Early Stopping to stop training if there is no significant improvement.
   - The trainer logs metrics like Accuracy, F1 Score, Precision, and Recall.

4. Model Saving:
   - After training, the fine-tuned model and tokenizer are saved.
   - The model is uploaded to Hugging Face Hub for easy sharing.

Metrics
During training, the following metrics are tracked:
- Accuracy: The proportion of correct predictions.
- F1 Score: A weighted average of Precision and Recall.
- Precision: The proportion of positive predictions that are actually correct.
- Recall: The proportion of actual positive cases that are correctly identified.

Results
After training, the model achieved the following performance:

| Step | Training Loss | Validation Loss | Accuracy | F1 Score | Precision | Recall |
|------|---------------|-----------------|----------|----------|-----------|--------|
| 25   | 1.546         | 0.437           | 0.295    | 0.248    | 0.437     | 0.437  |
| 50   | 0.681         | 0.748           | 0.714    | 0.782    | 0.748     | 0.748  |
| 75   | 0.532         | 0.828           | 0.781    | 0.765    | 0.828     | 0.828  |
| 100  | 0.303         | 0.912           | 0.900    | 0.915    | 0.912     | 0.912  |
| 125  | 0.158         | 0.962           | 0.958    | 0.956    | 0.962     | 0.962  |
| 150  | 0.084         | 0.978           | 0.974    | 0.971    | 0.978     | 0.978  |
| 175  | 0.078         | 0.985           | 0.982    | 0.978    | 0.986     | 0.986  |

Model Upload
Once training is complete, the model is saved and pushed to the Hugging Face Hub for easy access:
[ModernBERT-DDOS Model on Hugging Face](https://huggingface.co/ccaug/modernbert_ddos)

Download Model
You can download the model from Hugging Face using the model name ccaug/modernbert_ddos.
