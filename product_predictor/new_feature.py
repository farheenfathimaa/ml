import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify

# Load the base category data and model (assuming it's already trained)
base_data_path = "/home/farheenfathimaa/ml-projects/product_predictor/data/oio_category.csv"
base_test_data = pd.read_csv(base_data_path)
base_categories = base_test_data[['category_id', 'name']].drop_duplicates().reset_index(drop=True)
base_names = base_categories['name'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', str(x).lower()).strip()).tolist()
model1 = SentenceTransformer('all-MiniLM-L6-v2')
base_category_embeddings = model1.encode(base_names, convert_to_tensor=True)

class FeaturePredictor:
    def __init__(self, data_path="/home/farheenfathimaa/ml-projects/product_predictor/data"):
        self.data_path = data_path
        self.feature_dfs = {}
        self.feature_embeddings = {}
        self.load_features()

    def load_features(self):
        # Look for new feature CSVs in the data directory
        for filename in os.listdir(self.data_path):
            if filename.endswith(".csv") and "oio_category" not in filename:
                feature_name = os.path.splitext(filename)[0]
                feature_df = pd.read_csv(os.path.join(self.data_path, filename))
                
                # Assume the new CSV has columns 'category_id', 'name', and a new feature column
                if 'category_id' in feature_df.columns and 'name' in feature_df.columns:
                    feature_columns = [col for col in feature_df.columns if col not in ['category_id', 'name']]
                    if feature_columns:
                        for feature_col in feature_columns:
                            self.feature_dfs[feature_col] = feature_df
                            feature_contents = feature_df[feature_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', str(x).lower()).strip()).tolist()
                            self.feature_embeddings[feature_col] = model1.encode(feature_contents, convert_to_tensor=True)
                            print(f"Loaded new feature: '{feature_col}' from {filename}")

    def get_additional_features(self, category_name):
        results = {}
        for feature_name, feature_df in self.feature_dfs.items():
            # Find all feature entries for the given category name
            matching_entries = feature_df[feature_df['name'].str.lower() == category_name.lower()][feature_name].tolist()
            if matching_entries:
                results[feature_name] = matching_entries
            
        return results

# This function should be integrated into your main app logic.
def find_and_add_features(prediction_result, feature_predictor):
    # This is the main function that will be called after a successful category prediction
    category_name = prediction_result.get('category')
    if category_name and category_name != 'Unknown':
        additional_features = feature_predictor.get_additional_features(category_name)
        prediction_result['features'] = additional_features
    return prediction_result