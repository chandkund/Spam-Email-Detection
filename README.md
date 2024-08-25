# Spam Email Detection

This project aims to classify emails as either spam or not spam using a fine-tuned DistilBERT model. The dataset consists of email text with labels indicating whether each email is spam (1) or not spam (0). The model is trained using the Hugging Face Transformers library and leverages the DistilBERT architecture for efficient and accurate text classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Project Overview

This project uses DistilBERT, a distilled version of BERT, to detect spam emails. The dataset includes email texts and a binary label indicating whether each email is spam. The project involves data preprocessing, model training, and evaluation to achieve accurate spam detection.

## Installation

To run this project, you’ll need Python and the following libraries:

- `torch`
- `transformers`
- `pandas`
- `scikit-learn`

Install the required packages using `pip`:

```bash
pip install torch transformers pandas scikit-learn
```

## Usage

1. **Prepare the Dataset:**
   - Ensure your dataset is in a CSV file with columns labeled `text` and `spam`.

2. **Run the Code:**
   - Use the following code to preprocess data, train the model, and evaluate performance.

3. **Train the Model:**
   - Run the script to train the DistilBERT model on your dataset.

4. **Evaluate the Model:**
   - After training, evaluate the model using accuracy, precision, recall, and F1-score metrics.

## Code Explanation

### 1. Importing Libraries

```python
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
```

### 2. Load and Preprocess Data

```python
# Load dataset
df = pd.read_csv('spam_dataset.csv')

# Split into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['spam'], test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)
```

### 3. Define the Model

```python
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
```

### 4. Training the Model

```python
model.train()
for epoch in range(3):  # Set the number of epochs
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete! Loss: {loss.item()}")
```

### 5. Evaluate the Model

```python
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    }

```
## Model Evaluation

The trained DistilBERT model is evaluated on the test dataset using accuracy_score. These metrics provide insight into the model's performance in correctly identifying spam and non-spam emails.
```python

accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy:.4f}")
```
Validation Accuracy: 0.9939
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
