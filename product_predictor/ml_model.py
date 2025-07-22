# ml_model.py - YOUR EXACT CODE WITH MINOR PATH ADJUSTMENTS

import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import os

class ProductCategoryPredictor:
    def __init__(self, csv_path='data/oio_category.csv'):
        """Initialize the predictor with your exact logic"""
        
        # Load data (YOUR CODE - UNCHANGED)
        self.test_data = pd.read_csv(csv_path)
        
        # Extract relevant columns and clean data (YOUR CODE - UNCHANGED)
        self.categories = self.test_data[['category_id', 'name']].drop_duplicates().reset_index(drop=True)
        
        # Technical term corrections - only fix obvious misspellings (YOUR CODE - UNCHANGED)
        self.tech_corrections = {
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
        
        # Initialize model and embeddings
        self._setup_model()
    
    def _setup_model(self):
        """Setup the model and generate embeddings"""
        # Preprocess category names (YOUR CODE - UNCHANGED)
        test_names = self.categories['name'].apply(self.preprocess_text).tolist()
        
        # Load the pre-trained model (YOUR CODE - UNCHANGED)
        self.model1 = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings for all category names (YOUR CODE - UNCHANGED)
        self.category_embeddings = self.model1.encode(test_names, convert_to_tensor=True)
    
    # Simple text cleaning function (YOUR CODE - UNCHANGED)
    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text.strip()
    
    # New spelling correction function - only fixes technical terms (YOUR CODE - UNCHANGED)
    def correct_spelling(self, query):
        words = query.lower().split()
        corrected_words = []
        
        for word in words:
            if word in self.tech_corrections:
                corrected_words.append(self.tech_corrections[word])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    # Generate spelling suggestions - only for technical terms (YOUR CODE - UNCHANGED)
    def get_spelling_suggestions(self, query):
        words = query.lower().split()
        suggestions = []
        
        for word in words:
            if word in self.tech_corrections and word != self.tech_corrections[word]:
                suggestions.append(f"'{word}' might be '{self.tech_corrections[word]}'")
        
        return suggestions
    
    def predict_category(self, description):
        """YOUR EXACT PREDICTION FUNCTION - UNCHANGED"""
        try:
            original_query = description
            spelling_suggestions = self.get_spelling_suggestions(description)
            
            # Correct spelling using our technical terms dictionary
            corrected_query = self.correct_spelling(description)
            used_query = corrected_query if corrected_query != description else description
            
            # Preprocess and encode the input
            processed_text = self.preprocess_text(used_query)
            query_embedding = self.model1.encode(processed_text, convert_to_tensor=True)
            
            # Calculate similarity with all categories
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.category_embeddings.cpu().numpy()
            ).flatten()
            
            # Get the best match
            best_idx = similarities.argmax()
            best_score = similarities[best_idx]
            
            best_cat_id = self.categories.iloc[best_idx]['category_id']
            best_cat_name = self.categories.iloc[best_idx]['name']
            
            return {
                'original_query': original_query,
                'used_query': used_query,
                'synonym': used_query,
                'Predicted Cat ID': best_cat_id,
                'Predicted Cat Name': best_cat_name,
                'Confidence Score': float(best_score),
                'spelling_suggestions': spelling_suggestions
            }
        
        except Exception as e:
            print(f"Error processing '{description}': {str(e)}")
            return {
                'original_query': description,
                'used_query': description,
                'synonym': description,
                'Predicted Cat ID': None,
                'Predicted Cat Name': "Unknown",
                'Confidence Score': 0.0,
                'spelling_suggestions': []
            }
    
    def predict_multiple(self, descriptions):
        """Predict categories for multiple descriptions"""
        results = []
        for desc in descriptions:
            result = self.predict_category(desc.strip())
            results.append(result)
        return results

# Initialize global predictor instance
predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global predictor
    if predictor is None:
        predictor = ProductCategoryPredictor()
    return predictor
