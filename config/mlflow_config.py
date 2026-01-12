import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

class MLflowConfig:
    """Configuration centralisée pour MLflow"""
    
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
    TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    # Nom du modèle enregistré dans le Registry
    MODEL_NAME = os.getenv("MODEL_NAME", "mnist-classifier")
    
    # Seuils de promotion automatique
    MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.95"))
    MIN_F1_SCORE = float(os.getenv("MIN_F1_SCORE", "0.94"))
    
    # Configuration S3/MinIO pour artifacts
    S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    @classmethod
    def setup_mlflow(cls):
        """Configure les credentials MLflow"""
        
        mlflow.set_tracking_uri(cls.TRACKING_URI)
        
        if cls.TRACKING_USERNAME and cls.TRACKING_PASSWORD:
            os.environ['MLFLOW_TRACKING_USERNAME'] = cls.TRACKING_USERNAME
            os.environ['MLFLOW_TRACKING_PASSWORD'] = cls.TRACKING_PASSWORD
        
        if cls.S3_ENDPOINT_URL:
            os.environ['MLFLOW_S3_ENDPOINT_URL'] = cls.S3_ENDPOINT_URL
            os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
            os.environ['AWS_S3_ADDRESSING_STYLE'] = 'path'
        if cls.AWS_ACCESS_KEY_ID:
            os.environ['AWS_ACCESS_KEY_ID'] = cls.AWS_ACCESS_KEY_ID
        if cls.AWS_SECRET_ACCESS_KEY:
            os.environ['AWS_SECRET_ACCESS_KEY'] = cls.AWS_SECRET_ACCESS_KEY
        
        return mlflow