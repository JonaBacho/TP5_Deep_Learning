from flask import Flask, request, jsonify
import mlflow
import mlflow.tensorflow
import numpy as np
import os
from config.mlflow_config import MLflowConfig

app = Flask(__name__)

# Configuration MLflow
mlflow_client = MLflowConfig.setup_mlflow()

class ModelManager:
    """Gestionnaire de modèle avec chargement depuis MLflow"""
    
    def __init__(self):
        self.model = None
        self.model_version = None
        self.load_production_model()
    
    def load_production_model(self):
        """Charge le modèle marqué 'Production' depuis MLflow"""
        try:
            model_name = MLflowConfig.MODEL_NAME
            stage = "Production"
            
            print(f"Chargement du modèle '{model_name}' en stage '{stage}'...")
            
            # Charger le modèle depuis le Model Registry
            model_uri = f"models:/{model_name}/{stage}"
            self.model = mlflow.tensorflow.load_model(model_uri)
            
            # Récupérer les informations de version
            client = mlflow.MlflowClient()
            model_versions = client.get_latest_versions(model_name, stages=[stage])
            
            if model_versions:
                self.model_version = model_versions[0].version
                print(f"✓ Modèle chargé: version {self.model_version}")
            else:
                print("⚠ Aucun modèle en Production trouvé!")
                self.model = None
                
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {str(e)}")
            print("⚠ Tentative de chargement d'un modèle de fallback...")
            self.load_fallback_model()
    
    def load_fallback_model(self):
        """Charge un modèle de fallback local si disponible"""
        fallback_path = os.getenv("FALLBACK_MODEL_PATH", "models/mnist_model.h5")
        if os.path.exists(fallback_path):
            import tensorflow as tf
            self.model = tf.keras.models.load_model(fallback_path)
            self.model_version = "fallback"
            print(f"✓ Modèle de fallback chargé depuis {fallback_path}")
        else:
            print("❌ Aucun modèle de fallback disponible")
            self.model = None
    
    def reload_model(self):
        """Recharge le modèle depuis MLflow"""
        self.load_production_model()
    
    def predict(self, image_data):
        """Fait une prédiction"""
        if self.model is None:
            raise RuntimeError("Aucun modèle n'est chargé")
        
        prediction = self.model.predict(image_data)
        return prediction

# Initialiser le gestionnaire de modèle
model_manager = ModelManager()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de santé"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager.model is not None,
        'model_version': model_manager.model_version,
        'model_name': MLflowConfig.MODEL_NAME
    }), 200

@app.route('/model/info', methods=['GET'])
def model_info():
    """Informations sur le modèle actuel"""
    if model_manager.model is None:
        return jsonify({'error': 'No model loaded'}), 503
    
    return jsonify({
        'model_name': MLflowConfig.MODEL_NAME,
        'model_version': model_manager.model_version,
        'stage': 'Production',
        'mlflow_tracking_uri': MLflowConfig.TRACKING_URI
    }), 200

@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Recharge le modèle depuis MLflow (utile après une promotion)"""
    try:
        model_manager.reload_model()
        return jsonify({
            'message': 'Model reloaded successfully',
            'model_version': model_manager.model_version
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    
    if model_manager.model is None:
        return jsonify({'error': 'No model loaded'}), 503
    
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_data = np.array(data['image'])
        image_data = image_data.reshape(1, 784)
        image_data = image_data.astype("float32") / 255.0
        
        prediction = model_manager.predict(image_data)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': prediction.tolist(),
            'model_version': model_manager.model_version
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)