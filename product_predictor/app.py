# app.py
import logging
import logging.handlers
import json
import socket
import os
from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Add the current directory to the system path to ensure imports work
# This resolves the ModuleNotFoundError
sys.path.insert(0, os.path.dirname(__file__))

# Import the new_feature module now that the path is set
from new_feature import FeaturePredictor, find_and_add_features

# --- Logging Configuration for Logstash ---
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
            "pathname": record.pathname,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "process": record.process,
            "thread": record.thread,
        }
        if hasattr(record, 'extra_data'):
            log_record.update(record.extra_data)
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

class LogstashSocketHandler(logging.handlers.SocketHandler):
    def emit(self, record):
        try:
            msg = self.formatter.format(record) + '\n'
            self.send(msg.encode('utf-8'))
        except Exception:
            self.handleError(record)

LOGSTASH_HOST = os.getenv('LOGSTASH_HOST', 'localhost')
LOGSTASH_PORT = int(os.getenv('LOGSTASH_PORT', 5044))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

logstash_enabled = os.getenv('ENABLE_LOGSTASH', 'false').lower() == 'true'
if logstash_enabled:
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(2)
        result = test_socket.connect_ex((LOGSTASH_HOST, LOGSTASH_PORT))
        test_socket.close()
        if result == 0:
            logstash_handler = LogstashSocketHandler(LOGSTASH_HOST, LOGSTASH_PORT)
            logstash_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(logstash_handler)
            print(f"Logstash handler configured for {LOGSTASH_HOST}:{LOGSTASH_PORT}")
        else:
            print(f"Warning: Cannot connect to Logstash at {LOGSTASH_HOST}:{LOGSTASH_PORT}")
    except Exception as e:
        print(f"Warning: Could not connect to Logstash at {LOGSTASH_HOST}:{LOGSTASH_PORT}. Error: {e}")
else:
    print("Logstash logging disabled. Set ENABLE_LOGSTASH=true to enable.")

logger = logging.getLogger(__name__)

# --- Product Predictor Model Logic ---
try:
    # Use os.path.join for robust path handling
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    data_path = os.path.join(data_dir, "oio_category.csv")
    test_data = pd.read_csv(data_path)
    categories = test_data[['category_id', 'name']].drop_duplicates().reset_index(drop=True)

    tech_corrections = {
        'motr': 'motor', 'wasing': 'washing', 'machn': 'machine',
        'pmp': 'pump', 'dishwshr': 'dishwasher', 'blwr': 'blower',
        'heetr': 'heater', 'gyser': 'geyser', 'swich': 'switch',
        'grndr': 'grinder', 'compresr': 'compressor', 'sensr': 'sensor',
        'turbin': 'turbine', 'reley': 'relay', 'circut': 'circuit',
        'invertr': 'inverter', 'modul': 'module', 'solr': 'solar',
        'bearng': 'bearing', 'rollr': 'roller', 'actuatrr': 'actuator',
        'dampr': 'damper', 'driv': 'drive', 'extruuder': 'extruder',
        'couplng': 'coupling', 'bernr': 'burner', 'sytem': 'system',
        'ctrl': 'control', 'bord': 'board', 'un': 'unit', 'fl': 'unit'
    }

    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text.strip()

    def correct_spelling(query):
        words = query.lower().split()
        corrected_words = [tech_corrections.get(word, word) for word in words]
        return ' '.join(corrected_words)

    # Preprocess category names
    test_names = categories['name'].apply(preprocess_text).tolist()
    
    # Load the pre-trained model
    model1 = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for all category names
    category_embeddings = model1.encode(test_names, convert_to_tensor=True)

except Exception as e:
    logger.error(f"Failed to load ML model or data: {e}")
    sys.exit(1)

class ProductPredictor:
    def predict_single(self, description):
        try:
            original_query = description
            
            # Correct spelling using the technical terms dictionary
            corrected_query = correct_spelling(description)
            used_query = corrected_query if corrected_query != description else description

            # Preprocess and encode the input
            processed_text = preprocess_text(used_query)
            query_embedding = model1.encode(processed_text, convert_to_tensor=True)

            # Calculate similarity with all categories
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                category_embeddings.cpu().numpy()
            ).flatten()

            best_idx = similarities.argmax()
            best_score = similarities[best_idx]

            best_cat_id = categories.iloc[best_idx]['category_id']
            best_cat_name = categories.iloc[best_idx]['name']

            return {
                'category': best_cat_name,
                'confidence': f"{best_score:.2f}",
                'id': str(best_cat_id),
                'processed_query': used_query
            }
        except Exception as e:
            logger.error(f"Error during single prediction: {e}")
            return {
                'category': "Prediction Failed",
                'confidence': "0.00",
                'id': "N/A",
                'processed_query': description
            }

    def predict_multiple(self, descriptions):
        results = []
        for desc in descriptions:
            prediction = self.predict_single(desc)
            results.append(prediction)
        return results

predictor = ProductPredictor()
feature_predictor = FeaturePredictor()

# --- Flask Application Routes ---
app = Flask(__name__)

@app.route('/')
def home():
    try:
        return open("templates/index.html").read()
    except FileNotFoundError:
        return "<h1>Product Predictor API</h1><p>Use /predict_single or /predict_multiple endpoints</p>"

@app.route('/predict_single', methods=['POST'])
def predict_single_product():
    # Reload features on each request to check for file changes
    feature_predictor.load_features()
    
    data = request.json
    description = data.get('product_description')
    if not description:
        logger.error("Single product prediction: Missing 'product_description'")
        return jsonify({"error": "Missing product_description"}), 400

    prediction = predictor.predict_single(description)
    logger.info(f"Single product prediction result: {prediction}")
    
    # Only add features if they exist
    prediction = find_and_add_features(prediction, feature_predictor)
    
    return jsonify(prediction)

@app.route('/predict_multiple', methods=['POST'])
def predict_multiple_products():
    # Reload features on each request to check for file changes
    feature_predictor.load_features()
    
    data = request.json
    descriptions = data.get('product_descriptions')
    if not descriptions or not isinstance(descriptions, list):
        logger.error("Multiple product prediction: Missing or invalid 'product_descriptions'")
        return jsonify({"error": "Missing or invalid product_descriptions (expected a list)"}), 400

    predictions = predictor.predict_multiple(descriptions)
    
    # Corrected line to find and add the new features
    predictions = [find_and_add_features(p, feature_predictor) for p in predictions]
    
    logger.info(f"Multiple product predictions result: {predictions}")
    return jsonify(predictions)

if __name__ == "__main__":
    logger.info("Starting Product Predictor application...")
    app.run(debug=True, host='0.0.0.0', port=5000)