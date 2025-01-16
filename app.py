from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Initialize Flask app
app = Flask(__name__)

# Set the environment (default to production)
app.config['ENV'] = os.getenv('FLASK_ENV', 'production')

# Load the fine-tuned model and tokenizer
model_path = os.getenv('MODEL_PATH', "./finetuned_model")  # Use environment variable for model path
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Label mapping
label_mapping = {0: "false", 1: "true", 2: "misleading"}

# Define a route for the root URL
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the BERT Fine-tuned Model API!",
        "instructions": "Use the /predict endpoint with a POST request to classify text.",
        "environment": app.config['ENV']
    })

# Define route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON request
        data = request.get_json()
        text = data.get("text", None)
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Tokenize input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        ).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

        # Get predicted label and format probabilities
        predicted_label_index = probabilities.argmax()
        predicted_label = label_mapping[predicted_label_index]
        probabilities_percentage = (probabilities * 100).round(2).tolist()

        # Prepare response
        response = {
            "text": text,
            "predicted_label": predicted_label,
            "probabilities": {
                "false": probabilities_percentage[0],
                "true": probabilities_percentage[1],
                "misleading": probabilities_percentage[2]
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    # Host and port are configurable with environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug_mode = app.config['ENV'] == 'development'
    app.run(host=host, port=port, debug=debug_mode)
