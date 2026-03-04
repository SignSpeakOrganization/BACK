import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# --- 1. CONFIGURATION DU MODÈLE ---
model_path = 'hand_landmarker.task'

# --- 2. FONCTIONS DE DESSIN ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pouce
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Majeur
    (0, 13), (13, 14), (14, 15), (15, 16), # Annulaire
    (0, 17), (17, 18), (18, 19), (19, 20), # Auriculaire
    (5, 9), (9, 13), (13, 17)              # Paume
]

def draw_hand_landmarks_tasks_only(image_bgr, hand_landmarks_list):
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
        for (x, y) in pts:
            cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
    return annotated

# --- 3. CONFIGURATION DE LA CAPTURE DYNAMIQUE ---
action = input("Quel signe (mot) veux-tu enregistrer ? ").strip().lower()

try:
    nb_input = input("Combien d'essais veux-tu faire ? (Entrée = 30) : ")
    nb_sequences = int(nb_input) if nb_input else 30
except ValueError:
    nb_sequences = 30

# Choix de la caméra (0 = iPhone en général, 1 = Mac)
try:
    cam_input = input("Quelle caméra utiliser ? (0 = iPhone, 1 = Mac) [Entrée = 1] : ")
    cam_index = int(cam_input) if cam_input else 1
except ValueError:
    cam_index = 1

longueur_sequence = 30   # 30 images = environ 1 seconde de mouvement
data_path = os.path.join('dataset')
os.makedirs(os.path.join(data_path, action), exist_ok=True)

# Configuration de MediaPipe Tasks
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)

print(f"\n--- PRÉPARATION ---")
print(f"On va enregistrer {nb_sequences} fois le signe '{action}'.")

# --- 4. BOUCLE D'ENREGISTREMENT ---
with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(cam_index)

    for sequence in range(nb_sequences):
        
        # --- COMPTE À REBOURS VISUEL ---
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                # Affichage du texte au centre de l'écran
                cv2.putText(frame, f"GO DANS {i}...", (150, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
                cv2.putText(frame, f"Essai {sequence}/{nb_sequences} - {action}", (15, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Collecte LSF', frame)
                cv2.waitKey(1000) # Attendre 1 seconde par chiffre

        window = []
        
        # --- ENREGISTREMENT DU MOUVEMENT (30 FRAMES) ---
        for frame_num in range(longueur_sequence):
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            
            # Préparation de l'image pour l'API Tasks
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Détection
            result = landmarker.detect(mp_image)

            # Extraction des points
            keypoints = np.zeros(126) # 21 points * 3 (x,y,z) * 2 mains
            
            if result.hand_landmarks:
                frame = draw_hand_landmarks_tasks_only(frame, result.hand_landmarks)
                
                all_landmarks = []
                for hand in result.hand_landmarks:
                    for lm in hand:
                        all_landmarks.extend([lm.x, lm.y, lm.z])
                
                arr = np.array(all_landmarks)
                limit = min(len(arr), 126)
                keypoints[:limit] = arr[:limit]

            window.append(keypoints)

            # Affichage pendant l'enregistrement ("ENREGISTREMENT EN COURS")
            cv2.putText(frame, 'ENREGISTREMENT...', (150,250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, f'Essai {sequence}/{nb_sequences} - {action}', (15,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Collecte LSF', frame)

            if cv2.waitKey(10) & 0xFF == 27: # Echap pour quitter
                break

        # Sauvegarde
        npy_path = os.path.join(data_path, action, str(sequence))
        np.save(npy_path, np.array(window))
        print(f"Essai {sequence} sauvegardé.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTerminé ! Données sauvegardées dans dataset/{action}/")