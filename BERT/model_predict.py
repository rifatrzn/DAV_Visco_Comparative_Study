import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import matplotlib.pyplot as plt
import re
import seaborn as sns


# Assuming the paths to your trained model and tokenizer
model_path = './model_save'
tokenizer_path = './tokenizer_save'

# Load the new dataset
df_new = pd.read_csv('master_table.csv')

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# If CUDA is available, use GPU, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Map from predicted intent to visualization types
# Adjusted intent-to-visualization mapping based on JSON
intent_to_viz = {
    "distribution": "pie chart",
    "trend analysis": "line graph",
    "accumulation": "bar chart",
    "top entities ranking": "horizontal bar chart",
    "correlation analysis": "scatter plot",
    "category specific top entities ranking": "horizontal bar chart",
    # If "stacked bar chart" is another valid visualization, it should be mapped from a different intent
}

# Function to predict the intent of a query using the trained model
def predict_intent_with_confidence(query, tokenizer, model, device):
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_intent_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_intent_id].item()

    # Check if predicted_intent_id is within the range of intents defined
    if predicted_intent_id < len(intent_to_viz.keys()):
        predicted_intent = list(intent_to_viz.keys())[predicted_intent_id]
    else:
        # Handle the case where predicted_intent_id is out of range
        predicted_intent = "Unknown"
        confidence = 0  # Optionally, set to a very low confidence to indicate uncertainty

    return predicted_intent, confidence



# Take user query input and predict visualization suggestion
user_query = input("Please enter your query for a dashboard visualization suggestion: ")
predicted_intent, confidence = predict_intent_with_confidence(user_query, tokenizer, model, device)

# Obtain the visualization suggestion based on the predicted intent
visualization_suggestion = intent_to_viz.get(predicted_intent, "No suitable visualization found.")

# Outputting detailed analytics
print(f"Query: '{user_query}'")
print(f"Predicted intent: {predicted_intent} (Confidence: {confidence:.2f})")
print(f"Suggested visualization: {visualization_suggestion}")


# # Display the first few rows of the new dataframe to confirm it's loaded correctly
# print(df_new.head())



# Function to generate visualization based on predicted intent
# def generate_visualization(df, intent, entities):
#     if intent == "trend analysis":
#         # Assuming 'Year' and 'Quarter' are in your DataFrame if needed
#         df['Year'] = pd.DatetimeIndex(df['Invoice date']).year
#         df['Quarter'] = pd.DatetimeIndex(df['Invoice date']).quarter
#         sns.lineplot(data=df, x='Year', y='Amount line', estimator=sum, ci=None)
#         plt.title('Sales Trend Analysis')
#         plt.show()
#     elif intent == "distribution":
#         # Here's an example for distribution, adjust according to your data
#         df.groupby('Category')['Amount line'].sum().plot(kind='pie', autopct='%1.1f%%')
#         plt.title('Sales Distribution by Category')
#         plt.ylabel('')
#         plt.show()
#     elif intent == "top entities ranking":
#         # Extract city from 'Billing Addresses' assuming a simple format
#         # Adjust the split logic based on your actual address format
#         df['City'] = df['Billing Addresses'].apply(lambda x: x.split(',')[0])  # Simple extraction; customize as needed
#         top_entities = df.groupby('City')['Amount line'].sum().nlargest(5)
#         top_entities.plot(kind='bar')
#         plt.title('Top 5 Cities by Sales Revenue')
#         plt.xlabel('Total Sales Revenue')
#         plt.ylabel('City')
#         plt.tight_layout()  # Adjust layout for better readability
#         plt.show()
#     # Add other intents and corresponding visualization logic as needed


def extract_entities(query):
    entities = {}
    # Predefined patterns for different types of entities
    year_pattern = r'\b(20\d{2})\b'
    quarter_pattern = r'\bQ([1-4])\b'
    month_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b'
    category_pattern = r'category\s*[:=]\s*(\w+)'
    n_pattern = r'top\s*(\d+)'

    # Extract year
    year_match = re.search(year_pattern, query, re.IGNORECASE)
    if year_match:
        entities['Year'] = year_match.group(1)

    # Extract quarter
    quarter_match = re.search(quarter_pattern, query, re.IGNORECASE)
    if quarter_match:
        entities['Quarter'] = 'Q' + quarter_match.group(1)

    # Extract month
    month_match = re.search(month_pattern, query, re.IGNORECASE)
    if month_match:
        entities['Month'] = month_match.group(1)

    # Extract category
    category_match = re.search(category_pattern, query, re.IGNORECASE)
    if category_match:
        entities['Category'] = category_match.group(1).capitalize()  # Capitalize first letter

    # Extract N for ranking
    n_match = re.search(n_pattern, query, re.IGNORECASE)
    if n_match:
        entities['N'] = int(n_match.group(1))

    return entities


# Main workflow to include entity extraction

# Using the predicted intent and entities to generate a visualization
user_query = input("Please enter your query for a dashboard visualization suggestion: ")
predicted_intent, confidence = predict_intent_with_confidence(user_query, tokenizer, model, device)
entities = extract_entities(user_query)  # Extract entities from the user query

print(f"Query: '{user_query}'")
print(f"Predicted intent: {predicted_intent} (Confidence: {confidence:.2f})")

visualization_suggestion = intent_to_viz.get(predicted_intent, "No suitable visualization found.")
print(f"Suggested visualization: {visualization_suggestion}")

# Generate and show the visualization based on the predicted intent and extracted entities
# generate_visualization(df_new, predicted_intent, entities)