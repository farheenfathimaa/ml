# new_feature.py
import pandas as pd
import os

class FeaturePredictor:
    """
    Dynamically loads and predicts new features from CSV files in a specified directory.
    """
    def __init__(self):
        # The directory containing the feature CSVs
        self.features_dir = os.path.join(os.path.dirname(__file__), 'features')
        self.feature_data = {}
        self.load_features()

    def load_features(self):
        """
        Scans the features directory for CSVs and loads them into memory.
        This method is now called on every request to ensure the data is up-to-date.
        """
        self.feature_data = {}
        if not os.path.exists(self.features_dir):
            os.makedirs(self.features_dir)
            print(f"Created feature directory: {self.features_dir}")
            return

        for filename in os.listdir(self.features_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.features_dir, filename)
                try:
                    df = pd.read_csv(file_path)
                    if 'category_id' in df.columns:
                        feature_name = os.path.splitext(filename)[0]
                        # Store feature data keyed by category_id
                        self.feature_data[feature_name] = df.set_index('category_id').to_dict('index')
                        print(f"Loaded new feature '{feature_name}' from {filename}")
                    else:
                        print(f"Skipping {filename}: Missing 'category_id' column.")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        # If no files are found, self.feature_data remains an empty dictionary,
        # which effectively removes the feature functionality.

    def find_features_for_category(self, category_id):
        """
        Finds all new features for a given category ID.
        """
        category_id = int(category_id)
        features_for_id = {}
        for feature_name, data in self.feature_data.items():
            if category_id in data:
                # Add all attributes from the row
                features_for_id.update({k: v for k, v in data[category_id].items() if k != 'category_id'})
        return features_for_id

def find_and_add_features(prediction_result, feature_predictor_instance):
    """
    Appends new features to a single prediction result.
    This function is now called for every prediction, so it should be robust.
    """
    category_id = prediction_result.get('id')
    if category_id:
        features = feature_predictor_instance.find_features_for_category(category_id)
        # Only add the 'features' key if there are features to add
        if features:
            prediction_result['features'] = features
    return prediction_result