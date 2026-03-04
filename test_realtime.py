import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from tensorflow.keras.models import load_model

# --- 1. CHARGEMENT DU MODÈLE ET CONFIGURATION ---
model = load_model('modele_lsf.keras')
actions = np.array(['bonjour', 'neutre']) # L'ordre exact de ta Cellule 1 !
model_path = 'hand_landmarker.task'

# --- 2. CONFIGURATION DE MEDIAPIPE TASKS ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)

# Variables pour la détection en temps réel
sequence = [] # Va stocker les 30 dernières images (fenêtre glissante)
mot_actuel = "..."
seuil_confiance = 0.8 # L'IA doit être sûre à 80% pour afficher le mot

print("Lancement de la caméra... Appuie sur 'Echap' pour quitter.")

# --- 3. BOUCLE DE DÉTECTION ---
with vision.HandLandmarker.create_from_options(options) as landmarker:
    # Rappel : mets 0 si tu veux tester avec l'iPhone, ou 1 pour le Mac
    cap = cv2.VideoCapture(1) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Détection des mains
        result = landmarker.detect(mp_image)
        keypoints = np.zeros(126) 
        
        if result.hand_landmarks:
            all_landmarks = []
            for hand in result.hand_landmarks:
                for lm in hand:
                    all_landmarks.extend([lm.x, lm.y, lm.z])
            
            arr = np.array(all_landmarks)
            limit = min(len(arr), 126)
            keypoints[:limit] = arr[:limit]

        # On ajoute les points à notre séquence (notre mémoire à court terme)
        sequence.append(keypoints)
        
        # On garde uniquement les 30 dernières images (la fenêtre glissante)
        sequence = sequence[-30:]

        # Si on a bien 30 images en mémoire, on demande à l'IA !
        if len(sequence) == 30:
            # On donne la séquence à l'IA (le format doit être [1, 30, 126])
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            
            # Si l'IA est très sûre de sa réponse
            if res[np.argmax(res)] > seuil_confiance:
                mot_actuel = actions[np.argmax(res)]
            else:
                mot_actuel = "..."

        # Affichage du résultat sur la vidéo
        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, f"Prediction : {mot_actuel}", (15,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Test IA en direct', frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()