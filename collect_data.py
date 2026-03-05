import cv2
import numpy as np
import mediapipe as mp
import os

# Initialisation de MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- FONCTION D'EXTRACTION GLOBALE ---
def extraire_points_holistic(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

# --- CONFIGURATION ---
action = input("Quel signe (mot) veux-tu enregistrer ? ").strip().lower()

try:
    nb_sequences = int(input("Combien d'essais veux-tu faire ? (Entrée = 30) : ") or 30)
    cam_index = int(input("Quelle caméra utiliser ? (0 = iPhone, 1 = Mac) [Entrée = 1] : ") or 1)
except ValueError:
    nb_sequences = 30
    cam_index = 1

longueur_sequence = 30
dossier_action = os.path.join('dataset', action)
os.makedirs(dossier_action, exist_ok=True)

# --- CORRECTION : COMPTER LES FICHIERS EXISTANTS ---
# On compte combien d'essais sont déjà présents dans le dossier pour ne rien écraser
compteur_actuel = len(os.listdir(dossier_action))

print(f"\n--- PRÉPARATION ---")
print(f"Le dossier contient déjà {compteur_actuel} fichiers.")
print(f"On va enregistrer {nb_sequences} nouveaux essais pour le signe '{action}'.")

# --- BOUCLE D'ENREGISTREMENT ---
cap = cv2.VideoCapture(cam_index)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for sequence in range(nb_sequences):
        
        # --- COMPTE À REBOURS ---
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"GO DANS {i}...", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
                cv2.putText(frame, f"Essai {sequence+1}/{nb_sequences} - {action}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Collecte LSF Holistic', frame)
                cv2.waitKey(1000)

        window = []
        
        # --- ENREGISTREMENT (30 FRAMES) ---
        for frame_num in range(longueur_sequence):
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            # Extraction des 1662 points
            keypoints = extraire_points_holistic(results)
            window.append(keypoints)

            # Dessin Visuel
            image_rgb.flags.writeable = True
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.putText(frame, 'ENREGISTREMENT...', (150,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, f'Essai {sequence+1}/{nb_sequences} - {action}', (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Collecte LSF Holistic', frame)

            if cv2.waitKey(10) & 0xFF == 27: break

        # --- SAUVEGARDE SÉCURISÉE ---
        npy_path = os.path.join(dossier_action, str(compteur_actuel))
        np.save(npy_path, np.array(window))
        print(f"Essai {sequence+1}/{nb_sequences} sauvegardé sous le nom {compteur_actuel}.npy")
        
        # On incrémente le compteur global pour le prochain essai
        compteur_actuel += 1

cap.release()
cv2.destroyAllWindows()
print(f"\nTerminé ! Données sauvegardées dans {dossier_action}/")