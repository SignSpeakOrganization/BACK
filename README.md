# Projet de Traduction LSF (Langue des Signes Française) 🤟

Ce projet utilise l'intelligence artificielle (TensorFlow & MediaPipe) pour traduire la Langue des Signes en temps réel. Il est capable de détecter à la fois des signes statiques et des mots dynamiques en mouvement grâce à un réseau de neurones LSTM.

Contrairement aux approches classiques qui ne regardent que les mains, ce projet utilise **MediaPipe Holistic** pour extraire **1 662 points d'intérêt** par image (Visage, Buste, Épaules et Mains). Cela permet de comprendre la posture globale et les expressions faciales, qui sont fondamentales en LSF.

---

## 🛠️ 1. Mise en place et Installation (À faire une seule fois)

En raison de conflits de compatibilité très stricts entre les bibliothèques d'IA (notamment Numpy 2.0 et MediaPipe), **il est impératif d'installer les versions exactes ci-dessous**.

### Étape A : Créer et activer l'environnement virtuel
Ouvrez votre terminal et placez-vous dans le dossier du projet :

```bash
cd BACK
```

- **Activer l'environnement** :

  - **Mac** :

```bash
source venv/bin/activate
```

  - **Windows** :

```bash
.\venv\Scripts\activate
```

> Assurez-vous de toujours voir `(venv)` au début de votre ligne de commande avant de continuer.

### Étape B : Installation des dépendances

- **🍎 Mac (Processeurs Apple Silicon M1/M2/M3/M4)** : exécutez cette commande complète pour forcer les versions compatibles :

```bash
pip install "numpy<2.0" "opencv-python==4.9.0.80" "opencv-contrib-python==4.9.0.80" "jax==0.4.23" "jaxlib==0.4.23" "mediapipe==0.10.14" "tensorflow-macos==2.15.0" flask flask-cors jupyter matplotlib scikit-learn
```

- **🪟 Windows** : exécutez cette commande :

```bash
pip install "numpy<2.0" "opencv-python==4.9.0.80" "opencv-contrib-python==4.9.0.80" "mediapipe==0.10.14" "tensorflow==2.15.0" flask flask-cors jupyter matplotlib scikit-learn
```

## 🎥 2. Créer son Dataset (Apprendre un nouveau mot à l'IA)
L'IA a besoin de données pour apprendre. Un mouvement correspond à 30 images (frames) extraites sous forme de points mathématiques et sauvegardées dans des fichiers .npy.

Vous avez deux méthodes pour générer ces données :

### Méthode A : En direct avec la Webcam (Création manuelle)
Idéal si vous voulez faire le signe vous-même devant votre ordinateur.

Lancez le script :

```bash
python collect_data.py
```

Le terminal vous demandera le nom du mot (ex: bonjour), le nombre d'essais souhaités (ex: 30) et le numéro de la caméra (0 pour smartphone/cam externe, 1 pour webcam intégrée).

Un compte à rebours s'affiche à l'écran. Faites le signe ! Le script créera automatiquement un dossier dataset/bonjour/ rempli de vos fichiers d'entraînement.

### Méthode B : À partir d'une vidéo + Data Augmentation (Qualité Pro)
Idéal si vous avez une vidéo `.mp4` ou `.webm` de quelqu'un qui fait le signe. Cette méthode permet de multiplier artificiellement vos données pour rendre l'IA plus robuste.

Ouvrez le fichier `verify_and_extract.py` et modifiez les variables `VIDEO_PATH` et `MOT_LSF` au début du script pour cibler votre vidéo.

Lancez le script :

```bash
python verify_and_extract.py
```

La vidéo s'ouvre avec le squelette 3D dessiné sur la personne. Vérifiez que la détection de MediaPipe est bonne.

À la fin de la vidéo, le terminal vous demande si vous validez. Tapez o (oui).

Le script va générer automatiquement 7 variations mathématiques de cette seule vidéo (Originale, Bruitée, Zoom In, Zoom Out, Décalée, Rotation, Étirement) et les ajouter à votre Dataset.

## 🧠 3. Entraîner l'IA (Jupyter Notebook)
Une fois que vous avez généré vos données dans le dossier dataset/, il faut entraîner le modèle LSTM.

Lancez l'interface Jupyter :

```bash
jupyter notebook
```

Dans votre navigateur, ouvrez le fichier train_model.ipynb.

Assurez-vous d'utiliser le bon kernel lié à votre environnement virtuel.

> Attention : Vérifiez bien dans le bloc de création du modèle que la couche d'entrée attend 1662 valeurs (`input_shape=(30, 1662)`). Mettez également à jour le nombre de classes (mots) que vous entraînez.

Allez dans le menu Run et cliquez sur Run All Cells.

L'entraînement va démarrer. À la fin, deux graphiques s'afficheront :

- **Accuracy (Précision)** : les courbes doivent monter vers 1.0
- **Loss (Erreur)** : les courbes doivent descendre vers 0

Le script compile et sauvegarde automatiquement le fichier modele_lsf.keras. C'est le cerveau de votre application !

## 🚀 4. Lancer l'API Flask (Utilisation finale)
Maintenant que votre IA est entraînée, on lance le serveur web pour la connecter à l'interface utilisateur (Extension Chrome ou Site Web).

Lancez l'application :

```bash
python app.py
```

Le serveur tourne localement sur `http://localhost:5000`.

### Endpoints disponibles pour le Front-End

- `GET /start` : allume la webcam et lance la reconnaissance de l'IA en arrière-plan
- `GET /video_feed` : renvoie le flux vidéo traité de la caméra (utile pour afficher le retour vidéo dans l'interface utilisateur)
- `GET /sign` : renvoie la prédiction en temps réel au format JSON, par exemple :

```json
{
  "prediction": "bonjour",
  "mode": "DYNAMIQUE (Mot)"
}
```

- `GET /end` : coupe la caméra et arrête proprement le processus