import pytest
import numpy as np
from src.app import app, model_manager

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test du endpoint de santé"""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert data['status'] == 'healthy'

def test_model_info_endpoint(client):
    """Test du endpoint d'information du modèle"""
    response = client.get('/model/info')
    data = response.get_json()
    
    if model_manager.model is not None:
        assert response.status_code == 200
        assert 'model_name' in data
        assert 'model_version' in data
    else:
        assert response.status_code == 503

def test_predict_endpoint(client):
    """Test du endpoint de prédiction"""
    # Créer une image de test (28x28 = 784 pixels)
    test_image = np.random.rand(784).tolist()
    
    response = client.post('/predict',
                          json={'image': test_image},
                          content_type='application/json')
    
    if model_manager.model is not None:
        assert response.status_code == 200
        data = response.get_json()
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'probabilities' in data
        assert 0 <= data['prediction'] <= 9
    else:
        assert response.status_code == 503

def test_predict_invalid_input(client):
    """Test avec une entrée invalide"""
    response = client.post('/predict',
                          json={'wrong_key': []},
                          content_type='application/json')
    assert response.status_code == 400