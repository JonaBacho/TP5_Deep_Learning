import mlflow
from mlflow.tracking import MlflowClient
import argparse
from config.mlflow_config import MLflowConfig

# Configuration MLflow
mlflow_client = MLflowConfig.setup_mlflow()
client = MlflowClient()

def get_latest_model_version(model_name, stage=None):
    """Récupère la dernière version d'un modèle pour un stage donné"""
    if stage:
        versions = client.get_latest_versions(model_name, stages=[stage])
    else:
        versions = client.search_model_versions(f"name='{model_name}'")
        versions = sorted(versions, key=lambda x: int(x.version), reverse=True)
    
    return versions[0] if versions else None

def transition_model_stage(model_name, version, stage, archive_existing=True):
    """Change le stage d'un modèle"""
    print(f"Transition du modèle '{model_name}' version {version} vers '{stage}'...")
    
    # Si on passe en Production, archiver l'ancien modèle Production
    if stage == "Production" and archive_existing:
        current_prod = get_latest_model_version(model_name, "Production")
        if current_prod:
            print(f"Archivage de l'ancienne version Production: v{current_prod.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=current_prod.version,
                stage="Archived"
            )
    
    # Transition vers le nouveau stage
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    print(f"✓ Modèle '{model_name}' v{version} maintenant en '{stage}'")

def promote_to_production(model_name, version=None):
    """Promouvoir un modèle en Production"""
    if version is None:
        # Prendre la dernière version en Staging
        latest_staging = get_latest_model_version(model_name, "Staging")
        if latest_staging:
            version = latest_staging.version
        else:
            print("❌ Aucun modèle en Staging à promouvoir")
            return False
    
    transition_model_stage(model_name, version, "Production")
    return True

def promote_to_staging(model_name, version=None, run_id=None):
    """Promouvoir un modèle en Staging"""
    if version is None and run_id:
        # Trouver la version associée au run_id
        versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
        if versions:
            version = versions[0].version
        else:
            print(f"❌ Aucune version trouvée pour run_id: {run_id}")
            return False
    
    if version is None:
        # Prendre la dernière version sans stage
        versions = client.search_model_versions(f"name='{model_name}'")
        unassigned = [v for v in versions if v.current_stage == "None"]
        if unassigned:
            version = unassigned[0].version
        else:
            print("❌ Aucun modèle à promouvoir en Staging")
            return False
    
    transition_model_stage(model_name, version, "Staging", archive_existing=False)
    return True

def list_model_versions(model_name):
    """Liste toutes les versions d'un modèle"""
    versions = client.search_model_versions(f"name='{model_name}'")
    
    print(f"\nVersions du modèle '{model_name}':")
    print("-" * 80)
    
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        run = client.get_run(v.run_id)
        metrics = run.data.metrics
        
        print(f"Version: {v.version}")
        print(f"  Stage: {v.current_stage}")
        print(f"  Run ID: {v.run_id}")
        print(f"  Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
        print(f"  F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
        print(f"  Created: {v.creation_timestamp}")
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Gestion des stages de modèles MLflow")
    parser.add_argument("action", choices=["list", "staging", "production", "rollback"],
                       help="Action à effectuer")
    parser.add_argument("--model-name", default=MLflowConfig.MODEL_NAME,
                       help="Nom du modèle")
    parser.add_argument("--version", type=int, help="Version spécifique du modèle")
    parser.add_argument("--run-id", help="Run ID du modèle")
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_model_versions(args.model_name)
    
    elif args.action == "staging":
        promote_to_staging(args.model_name, args.version, args.run_id)
    
    elif args.action == "production":
        promote_to_production(args.model_name, args.version)
    
    elif args.action == "rollback":
        # Rollback = promouvoir une ancienne version archivée en Production
        if args.version:
            transition_model_stage(args.model_name, args.version, "Production")
        else:
            print("❌ Spécifiez --version pour le rollback")

if __name__ == "__main__":
    main()