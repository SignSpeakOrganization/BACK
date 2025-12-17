# Utiliser une image Python avec la version spécifiée
ARG PYTHON_VERSION=3.10.7
FROM python:${PYTHON_VERSION}

# Définir le dossier de travail
WORKDIR /app

# Installer les dépendances système nécessaires pour OpenCV et MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement requirements.txt d'abord (pour optimiser le cache Docker)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le reste du code source
COPY . .

# Exposer le port Flask
EXPOSE 5000

# Commande pour lancer l'application
CMD ["python", "app.py"]
