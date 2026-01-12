import mlflow
from mlflow.tracking import MlflowClient
from config.mlflow_config import MLflowConfig

# Configuration MLflow
mlflow_client = MLflowConfig.setup_mlflow()
client = MlflowClient()

def check_promotion_criteria(run_id):
    """Vérifie si un modèle remplit les critères de promotion"""
    run = client.get_run(run_id)
    metrics = run.data.metrics
    
    accuracy = metrics.get('test_accuracy', 0)
    f1_score = metrics.get('f1_score', 0)
    
    print(f"Vérification des critères pour Run ID: {run_id}")
    print(f"  Accuracy: {accuracy:.4f} (min: {MLflowConfig.MIN_ACCURACY})")
    print(f"  F1 Score: {f1_score:.4f} (min: {MLflowConfig.MIN_F1_SCORE})")
    
    meets_criteria = (
        accuracy >= MLflowConfig.MIN_ACCURACY and
        f1_score >= MLflowConfig.MIN_F1_SCORE
    )
    
    return meets_criteria, accuracy, f1_score

def auto_promote_model(run_id, model_name=None):
    """Promotion automatique basée sur les métriques"""
    
    if model_name is None:
        model_name = MLflowConfig.MODEL_NAME
    
    # Vérifier les critères
    meets_criteria, accuracy, f1_score = check_promotion_criteria(run_id)
    
    if not meets_criteria:
        print(f"❌ Le modèle ne remplit pas les critères de promotion")
        return False
    
    print(f"✓ Le modèle remplit les critères de promotion!")
    
    # Trouver la version du modèle
    versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
    
    if not versions:
        print(f"❌ Aucune version trouvée pour run_id: {run_id}")
        return False
    
    version = versions[0].version
    
    # Promouvoir en Staging d'abord
    print(f"Promotion en Staging...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )
    
    # Comparer avec le modèle actuel en Production
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    
    if prod_versions:
        prod_version = prod_versions[0]
        prod_run = client.get_run(prod_version.run_id)
        prod_accuracy = prod_run.data.metrics.get('test_accuracy', 0)
        
        print(f"\nComparaison avec Production:")
        print(f"  Production - Accuracy: {prod_accuracy:.4f}")
        print(f"  Nouveau    - Accuracy: {accuracy:.4f}")
        
        # Promouvoir seulement si meilleur que Production
        if accuracy > prod_accuracy:
            print(f"✓ Nouveau modèle meilleur → Promotion en Production!")
            
            # Archiver l'ancien Production
            client.transition_model_version_stage(
                name=model_name,
                version=prod_version.version,
                stage="Archived"
            )
            
            # Promouvoir en Production
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            
            print(f"✓ Modèle v{version} maintenant en Production!")
            return True
        else:
            print(f"⚠ Nouveau modèle pas meilleur → Reste en Staging")
            return False
    else:
        # Pas de modèle en Production, promouvoir directement
        print(f"✓ Pas de modèle en Production → Promotion automatique!")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        print(f"✓ Modèle v{version} maintenant en Production!")
        return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python auto_promote.py <run_id>")
        sys.exit(1)
    
    run_id = sys.argv[1]
    auto_promote_model(run_id)