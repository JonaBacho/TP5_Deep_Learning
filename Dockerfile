# Dockerfile
# Utiliser une image Python slim
FROM python:3.10-slim

# Définir les arguments de build
ARG PYTHON_VERSION=3.10

# Variables d'environnement
#ENV PYTHONUNBUFFERED=1 \
#    PYTHONDONTWRITEBYTECODE=1 \
#    PIP_NO_CACHE_DIR=1

# Créer un utilisateur non-root
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements et installer les dépendances
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

# Créer les répertoires nécessaires
RUN mkdir -p /app/models /app/logs && \
    chown -R appuser:appuser /app/models /app/logs

# Basculer vers l'utilisateur non-root
USER appuser

# Exposer le port
EXPOSE 5000

# Commande de démarrage avec gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "src.app:app"]
