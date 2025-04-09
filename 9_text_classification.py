# Install Required Libraries
# !pip install transformers datasets torch scikit-learn pandas


# Import necessary libraries
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.optim import AdamW  # Fixed import


# Load Pretrained BERT Model & Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


# Load and Prepare Data (IMDb Dataset)
df = pd.read_csv("imdb.csv")


# Check dataset structure
print(df.head())
print("Column names:", df.columns.tolist())


# Define text and label columns
text_column = 'Overview'  # Column containing movie reviews
label_column = 'IMDB_Rating'  # Column for IMDb ratings


# Convert IMDB_Rating to binary sentiment labels (positive: 1, negative: 0)
df[label_column] = df[label_column].apply(lambda x: 1 if x >= 7 else 0)


# Custom Dataset Class
class IMDbDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        item = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in item.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item


# Split dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df[text_column].tolist(),
    df[label_column].tolist(),
    test_size=0.2,
    random_state=42
)


# Create training and testing datasets
train_dataset = IMDbDataset(train_texts, train_labels)
test_dataset = IMDbDataset(test_texts, test_labels)


# Define DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# Define optimizer and move model to GPU if available
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Training loop (single epoch for simplicity)
model.train()
for batch in train_dataloader:
    inputs = {key: val.to(device) for key, val in batch.items() if key in ["input_ids", "attention_mask"]}
    labels = batch["label"].to(device)
   
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
   
    loss.backward()
    optimizer.step()
   
    print(f"Loss: {loss.item():.4f}")


# Evaluate the Model on the Test Set
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_dataloader:
        inputs = {key: val.to(device) for key, val in batch.items() if key in ["input_ids", "attention_mask"]}
        labels = batch["label"].to(device)
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())


# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")