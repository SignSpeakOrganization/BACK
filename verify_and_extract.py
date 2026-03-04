import cv2
import numpy as np
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
VIDEO_PATH = 'dataset/videos/bonjour.webm'  # Le chemin de ta vidéo à analyser
MOT_LSF = 'bonjour'      # Le dossier cible
MODEL_PATH_MP = 'hand_landmarker.task'

# ==========================================
# 🛠️ FONCTIONS DE DESSIN ET D'AUGMENTATION
# ==========================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), 
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]

def draw_landmarks(image_bgr, hand_landmarks_list):
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
        for (x, y) in pts:
            cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
    return annotated

def generer_variations(sequence_de_base):
    variations = []
    variations.append(sequence_de_base.copy()) # 1. Originale
    
    seq_3d = sequence_de_base.reshape((30, 42, 3))
    
    # 2. BRUIT (Tremblement)
    bruit = np.random.normal(0, 0.005, seq_3d.shape)
    variations.append((seq_3d + bruit).reshape((30, 126)))
    
    # 3. ZOOM IN
    seq_zoom_in = seq_3d.copy()
    seq_zoom_in[:, :, :2] = (seq_zoom_in[:, :, :2] - 0.5) * 1.20 + 0.5 
    variations.append(seq_zoom_in.reshape((30, 126)))
    
    # 4. ZOOM OUT
    seq_zoom_out = seq_3d.copy()
    seq_zoom_out[:, :, :2] = (seq_zoom_out[:, :, :2] - 0.5) * 0.85 + 0.5 
    variations.append(seq_zoom_out.reshape((30, 126)))
    
    # 5. DÉCALAGE
    seq_decalee = seq_3d.copy()
    seq_decalee[:, :, 0] += 0.06 # Décalage droite
    seq_decalee[:, :, 1] -= 0.04 # Décalage haut
    variations.append(seq_decalee.reshape((30, 126)))
    
    for var in variations:
        var[sequence_de_base == 0] = 0 # Protection contre les zéros
        
    return variations

# ==========================================
# 🚀 LECTURE ET VÉRIFICATION VISUELLE
# ==========================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH_MP)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)

print(f"\n--- Vérification de la vidéo : {VIDEO_PATH} ---")
sequence = []
cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
delay = int(1000 / fps) if fps > 0 else 30

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        result = landmarker.detect(mp_image)
        keypoints = np.zeros(126)
        
        if result.hand_landmarks:
            frame = draw_landmarks(frame, result.hand_landmarks)
            all_landmarks = []
            for hand in result.hand_landmarks:
                for lm in hand:
                    all_landmarks.extend([lm.x, lm.y, lm.z])
            arr = np.array(all_landmarks)
            limit = min(len(arr), 126)
            keypoints[:limit] = arr[:limit]

        sequence.append(keypoints)

        cv2.putText(frame, "VERIFICATION LANDMARKS", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Verification Video', frame)

        if cv2.waitKey(delay) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 💾 PRISE DE DÉCISION ET AUGMENTATION
# ==========================================
print("\n" + "="*50)
reponse = input("▶ Les landmarks étaient-ils bien placés ? Taper 'o' pour OUI (sauvegarder les 5 variations), 'n' pour NON : ")

if reponse.lower() == 'o':
    os.makedirs(os.path.join('dataset', MOT_LSF), exist_ok=True)
    compteur_actuel = len(os.listdir(os.path.join('dataset', MOT_LSF)))
    
    # Formatage à 30 images
    if len(sequence) >= 30:
        milieu = len(sequence) // 2
        sequence_base = sequence[milieu - 15 : milieu + 15]
    else:
        sequence_base = sequence + [np.zeros(126)] * (30 - len(sequence))
    
    sequence_base = np.array(sequence_base)
    
    print("\nTransformation mathématique des mouvements en cours...")
    cinq_variations = generer_variations(sequence_base)
    noms_variations = ["Originale", "Bruitée (Tremblement)", "Zoom In", "Zoom Out", "Décalée"]

    print(f"\n✅ Génération des fichiers à partir de l'index {compteur_actuel} :")
    for i, seq_aug in enumerate(cinq_variations):
        nom_fichier = f"{compteur_actuel}.npy"
        chemin = os.path.join('dataset', MOT_LSF, nom_fichier)
        np.save(chemin, seq_aug)
        print(f"  - Création de {nom_fichier} (Version : {noms_variations[i]})")
        compteur_actuel += 1

    print(f"\n🎉 Terminé ! Ton dossier '{MOT_LSF}' contient maintenant {compteur_actuel} fichiers.")
else:
    print("❌ Sauvegarde annulée. Le dataset n'a pas été modifié.")