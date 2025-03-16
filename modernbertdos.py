# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the trained model and tokenizer from Hugging Face Hub
model_name = "ccaug/modernbert_ddos"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset from command-line argument
import sys
if len(sys.argv) < 2:
    print("Usage: python script.py <dataset.csv>")
    sys.exit(1)

dataset_path = sys.argv[1]
df = pd.read_csv(dataset_path, low_memory=False)

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Define the desired labels
desired_labels = ['NetBIOS', 'BENIGN', 'LDAP', 'Portmap', 'Syn', 'MSSQL', 'UDP', 'SSL']

# Keep only the desired labels
df = df[df['Label'].isin(desired_labels)]

# Convert categorical labels to numerical labels
label_mapping = {label: i for i, label in enumerate(desired_labels)}
df['label'] = df['Label'].map(label_mapping)

# Prepare text data (concatenate features as text input)
def convert_to_text(row):
    return " ".join(map(str, row))

df['text'] = df.drop(columns=['Label', 'label']).apply(convert_to_text, axis=1)

# Use the entire dataset as the test set
test_data = df.copy()

# Convert to Hugging Face Dataset format
test_dataset = Dataset.from_pandas(test_data[['text', 'label']])

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

test_dataset = test_dataset.map(tokenize_function, batched=True)

# Prediction function
def predict_batch(batch):
    inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_labels = logits.argmax(dim=-1).cpu().numpy()
    return {"predictions": predicted_labels}

# Apply prediction with batch processing
predictions = test_dataset.map(predict_batch, batched=True, batch_size=6)

# Extract predicted and true labels
pred_labels = predictions['predictions']
true_labels = predictions['label']

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='weighted')
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Display results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=desired_labels))

print("\nConfusion Matrix:")
print(conf_matrix)
