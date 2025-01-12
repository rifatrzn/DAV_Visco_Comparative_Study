from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import spacy

# Load JSON file
with open('training_data.json', 'r') as file:
    data = json.load(file)

# Convert JSON to DataFrame
df = pd.json_normalize(data)

# Save the DataFrame to CSV
df.to_csv('training_data.csv', index=False)


# Encode the intents
label_encoder = LabelEncoder ()
df['intent_encoded'] = label_encoder.fit_transform (df['intent'])


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(query):
    # Process query with spaCy
    doc = nlp(query)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities


# Tokenization
tokenizer = BertTokenizer.from_pretrained ('bert-base-uncased')

# Tokenize all queries in the dataset
input_ids = []
attention_masks = []

for query in df['query']:
    # Extract entities
    entities = extract_entities(query)
    # Combine query with entity information (simple example)
    augmented_query = query + " " + " ".join([f"{ent[1]}" for ent in entities])
    # Proceed with tokenization as before using `augmented_query`
    encoded_dict = tokenizer.encode_plus (
        augmented_query, # Use augmented query,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=64,  # Pad & truncate all sentences
        padding='max_length',  # Pad all to the max length of the model
        truncation=True,  # Explicitly truncate to max length
        return_attention_mask=True,  # Construct attention masks
        return_tensors='pt',  # Return PyTorch tensors
    )

    input_ids.append (encoded_dict['input_ids'])
    attention_masks.append (encoded_dict['attention_mask'])

input_ids = torch.cat (input_ids, dim=0)
attention_masks = torch.cat (attention_masks, dim=0)
labels = torch.tensor(df['intent_encoded'].values).long()  # Ensure labels are Long type

# Split data into train and validation sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split (input_ids, labels,
                                                                                     random_state=2020, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split (attention_masks, labels, random_state=2020, test_size=0.1)

# Create the DataLoader
train_data = TensorDataset (train_inputs, train_masks, train_labels)
train_sampler = RandomSampler (train_data)
train_dataloader = DataLoader (train_data, sampler=train_sampler, batch_size=32)

validation_data = TensorDataset (validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler (validation_data)
validation_dataloader = DataLoader (validation_data, sampler=validation_sampler, batch_size=32)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_),  # Make sure this matches your dataset
    output_attentions=False,
    output_hidden_states=False,
)


import numpy as np


def flat_accuracy(preds, labels):
    """
    Function to calculate the accuracy of our predictions vs labels
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Training steps...

# Assuming you're using a GPU
device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
model.to (device)

# Define Optimizer and Scheduler
optimizer = AdamW (model.parameters (),
                   lr=2e-5,  # args.learning_rate
                   eps=1e-8)  # args.adam_epsilon

epochs = 4
total_steps = len (train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup (optimizer,
                                             num_warmup_steps=0,
                                             num_training_steps=total_steps)

# Training Loop
for epoch_i in range (0, epochs):
    print (f'======== Epoch {epoch_i + 1} / {epochs} ========')
    total_train_loss = 0

    # Evaluate the model on the validation set
    model.eval ()

    # Tracking variables
    total_eval_accuracy = 0

    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)  # Move batch to GPU
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        total_train_loss += loss.item()

        with torch.no_grad ():
            outputs = model (b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs.logits
        logits = logits.detach ().cpu ().numpy ()
        label_ids = b_labels.to ('cpu').numpy ()

        total_eval_accuracy += flat_accuracy (logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len (validation_dataloader)
    print (f"Validation Accuracy: {avg_val_accuracy:.2f}")

    # Validation step can be added here

print ("Training complete.")

model_save_path = "./model_save"
tokenizer_save_path = "./tokenizer_save"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)


from sklearn.metrics import precision_score, recall_score, f1_score

# Example calculation after prediction
# Convert logits to predictions for evaluation metrics
predictions = np.argmax(logits, axis=1)
precision = precision_score(label_ids, predictions, average='weighted')
recall = recall_score(label_ids, predictions, average='weighted')
f1 = f1_score(label_ids, predictions, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(label_ids, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
