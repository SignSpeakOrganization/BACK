# Utiliser une image Python avec la version spécifiée
ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}

# Définir le dossier de travail
WORKDIR /app

# Copier le code source de l'application
COPY . .

# Installer les dépendances
RUN pip install --upgrade pip
RUN pip install mediapipe
RUN pip install opencv-python
RUN pip install tensorflow
RUN pip install tf-nightly
RUN pip install scikit-learn
RUN pip install matplotlib

# Exposer les ports (si besoin pour Flask, FastAPI, etc.)
EXPOSE 5000

# Commande pour lancer l'application (modifie selon ton projet)
CMD ["python", "app.py"]
