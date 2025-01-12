import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import matplotlib.pyplot as plt

# Assuming paths to the tokenizer, model, and dataset
tokenizer_path = 'tokenizer_save'
model_path = 'model_save'
dataset_path = 'master_table.csv'


# Load dataset
class DataQuerier:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.dataset['date'] = pd.to_datetime(self.dataset['date'])

    def query_data(self, intent, entities):
        # Simplified example: Adjust based on your actual data columns and intent/entity structure
        if intent == "trend analysis" and 'year' in entities:
            filtered_data = self.dataset[self.dataset['date'].dt.year == int(entities['year'])]
            return filtered_data
        # Add more conditions for other intents and entities
        return pd.DataFrame()


# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


# Function to predict intent
def predict_intent(query, tokenizer, model, device):
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return prediction


# Visualization function (simplified example)
def generate_visualization(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], data['metric'])  # Assuming 'metric' is a column in your dataset
    plt.title('Trend Analysis')
    plt.xlabel('Date')
    plt.ylabel('Metric')
    plt.show()


# Main process
def main():
    querier = DataQuerier(dataset_path)

    user_query = input("Please enter your query for a dashboard visualization suggestion: ")
    predicted_intent = predict_intent(user_query, tokenizer, model, device)

    # Simplified: direct mapping of numeric predictions to intents and hardcoded entities extraction
    # In a real scenario, use a more sophisticated method for intent to text mapping and entity extraction
    intent_to_text = {0: "trend analysis"}  # Example mapping, expand based on your model
    entities = {'year': '2020'}  # Example extracted entities, implement extraction based on your needs

    if predicted_intent in intent_to_text:
        intent_text = intent_to_text[predicted_intent]
        filtered_data = querier.query_data(intent_text, entities)
        if not filtered_data.empty:
            generate_visualization(filtered_data)
        else:
            print("No data matches your query.")
    else:
        print("Intent not recognized.")


if __name__ == "__main__":
    main()
