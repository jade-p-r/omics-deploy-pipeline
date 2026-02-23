import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
from google.cloud import storage
from scripts.gene_preprocessor import GenePreprocessor

app = Flask(__name__)

def download_from_gcs(bucket_name, blob_name, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def load_artifacts():
    bucket_name = "biology-predict-bucket"
    
    download_from_gcs(bucket_name, "pca_random_forest_model.joblib", "/tmp/pca_random_forest_model.joblib")
    download_from_gcs(bucket_name, "pca_preprocessor.joblib", "/tmp/pca_preprocessor.joblib")
    
    model = joblib.load("/tmp/pca_random_forest_model.joblib")
    preprocessor = joblib.load("/tmp/pca_preprocessor.joblib")
    return model, preprocessor

model, preprocessor = load_artifacts()

def _predict_from_features(features):
    feature_array = np.asarray(features, dtype=float)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(1, -1)
    return model.predict(feature_array)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    
    #unwrap Vertex AI instances format
    if 'instances' in data:
        data = data['instances'][0]
    
    try:
        if 'gene_entry' in data:
            transformed = preprocessor.transform_new_gene_entry(data['gene_entry'])
            prediction = model.predict(transformed)
        elif 'features' in data:
            prediction = _predict_from_features(data['features'])
        else:
            return jsonify({
                'error': "Invalid payload. Provide either 'gene_entry' (raw gene values) or 'features' (preprocessed vector)."
            }), 400
    except (ValueError, TypeError) as exc:
        return jsonify({'error': str(exc)}), 400
    
    return jsonify({'predictions': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)