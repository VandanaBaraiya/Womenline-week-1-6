


!pip install flask-ngrok

!pip install flask

!pip install scikit-learn

!pip install pandas

import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok

from google.colab import files
import joblib

# Step 1: Upload intent_classifier.pkl
uploaded = files.upload()  # Click "Browse" and upload intent_classifier.pkl

# Load model using joblib
model = joblib.load('intent_classifier.pkl')

from google.colab import files
import joblib

# Step 2: Upload vectorizer.pkl
uploaded = files.upload()  # Click "Browse" and upload vectorizer.pkl

# Load vectorizer using joblib
vectorizer = joblib.load('vectorizer.pkl')

from google.colab import files
import pandas as pd

# Step 3: Upload legal_dataset.csv
uploaded = files.upload()  # Click "Browse" and upload legal_dataset.csv

# Load CSV dataset
dataset = pd.read_json('legal_dataset.json')

# Optional: Preview first few rows
print(dataset.head())

def get_response(user_query):
    input_vector = vectorizer.transform([user_query])
    predicted_intent = model.predict(input_vector)[0]

    matched_row = dataset[dataset['intent'] == predicted_intent]

    if not matched_row.empty:
        return matched_row.iloc[0]['response']
    else:
        return "Sorry, I couldn't find a suitable answer. Please rephrase your question."

from flask_ngrok import run_with_ngrok
run_with_ngrok(app)
app.run()

# legal_chatbot_backend.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained model, vectorizer, and dataset
model = joblib.load('intent_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')
dataset = pd.read_json('legal_dataset.json')  # Use read_csv() if file is .csv

from flask import Flask
app = Flask(__name__)

# Function to get chatbot response
def get_response(user_query):
    input_vector = vectorizer.transform([user_query])
    predicted_intent = model.predict(input_vector)[0]

    matched_row = dataset[dataset['intent'] == predicted_intent]

    if not matched_row.empty:
        return matched_row.iloc[0]['response']
    else:
        return "Sorry, I couldn't find a suitable answer. Please rephrase your question."

@app.route('/get_legal_answer', methods=['POST'])
def get_legal():   # ← naam change kar diya
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    response = get_response(user_query)
    return jsonify({"answer": response})

# Run app (for local testing)
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

app.run()