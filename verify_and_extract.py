import cv2
import numpy as np
import mediapipe as mp
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
VIDEO_PATH = 'videos/ça va.webm'  # Le chemin de ta vidéo à analyser
MOT_LSF = 'ça va'                         # Le dossier cible

# Initialisation de MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 🧠 EXTRACTION & DATA AUGMENTATION (1662 points)
# ==========================================
def extraire_points_holistic(results):
    # 1. Pose/Corps (33 points * 4 valeurs = 132)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    # 2. Visage (468 points * 3 valeurs = 1404)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    # 3. Main Gauche (21 points * 3 valeurs = 63)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    # 4. Main Droite (21 points * 3 valeurs = 63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    return np.concatenate([pose, face, lh, rh])

def appliquer_transformation(sequence, scale_x=1.0, scale_y=1.0, shift_x=0.0, shift_y=0.0, angle_deg=0.0):
    seq_trans = sequence.copy()
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    for f in range(len(seq_trans)):
        if np.all(seq_trans[f] == 0): continue # Ignore les frames vides
        
        # Transformation de la Pose (indices 0 à 131, format x, y, z, v)
        for i in range(0, 132, 4):
            nx, ny = seq_trans[f, i] - 0.5, seq_trans[f, i+1] - 0.5
            rx, ry = nx * c - ny * s, nx * s + ny * c
            seq_trans[f, i] = (rx * scale_x) + 0.5 + shift_x
            seq_trans[f, i+1] = (ry * scale_y) + 0.5 + shift_y

        # Transformation Visage + Mains (indices 132 à 1661, format x, y, z)
        for i in range(132, 1662, 3):
            if seq_trans[f, i] == 0 and seq_trans[f, i+1] == 0: continue
            nx, ny = seq_trans[f, i] - 0.5, seq_trans[f, i+1] - 0.5
            rx, ry = nx * c - ny * s, nx * s + ny * c
            seq_trans[f, i] = (rx * scale_x) + 0.5 + shift_x
            seq_trans[f, i+1] = (ry * scale_y) + 0.5 + shift_y

    return seq_trans

def generer_variations(sequence_de_base):
    variations = []
    variations.append(sequence_de_base.copy()) # 1. Originale
    
    # 2. BRUIT (Tremblement)
    bruit = np.random.normal(0, 0.003, sequence_de_base.shape)
    seq_bruitee = sequence_de_base + bruit
    seq_bruitee[sequence_de_base == 0] = 0 # Protection des zéros
    variations.append(seq_bruitee)
    
    # Utilisation de notre fonction mathématique pour les autres
    variations.append(appliquer_transformation(sequence_de_base, scale_x=1.2, scale_y=1.2)) # 3. Zoom In
    variations.append(appliquer_transformation(sequence_de_base, scale_x=0.85, scale_y=0.85)) # 4. Zoom Out
    variations.append(appliquer_transformation(sequence_de_base, shift_x=0.06, shift_y=-0.04)) # 5. Décalage
    variations.append(appliquer_transformation(sequence_de_base, angle_deg=10)) # 6. Rotation
    variations.append(appliquer_transformation(sequence_de_base, scale_x=1.3, scale_y=1.0)) # 7. Stretch Horizontal
    
    return variations

# ==========================================
# 🚀 LECTURE ET VÉRIFICATION VISUELLE
# ==========================================
print(f"\n--- Vérification de la vidéo : {VIDEO_PATH} ---")
sequence = []
cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
delay = int(1000 / fps) if fps > 0 else 30

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        
        # Extraction
        keypoints = extraire_points_holistic(results)
        sequence.append(keypoints)

        # Dessin Visuel
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.putText(frame, "VERIFICATION HOLISTIC", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Verification Video', frame)

        if cv2.waitKey(delay) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 💾 PRISE DE DÉCISION ET AUGMENTATION
# ==========================================
print("\n" + "="*50)
reponse = input("▶ Les landmarks étaient-ils bien placés ? Taper 'o' pour OUI (sauvegarder 7 variations), 'n' pour NON : ")

if reponse.lower() == 'o':
    os.makedirs(os.path.join('dataset', MOT_LSF), exist_ok=True)
    compteur_actuel = len(os.listdir(os.path.join('dataset', MOT_LSF)))
    
    if len(sequence) >= 30:
        milieu = len(sequence) // 2
        sequence_base = sequence[milieu - 15 : milieu + 15]
    else:
        sequence_base = sequence + [np.zeros(1662)] * (30 - len(sequence))
    
    sequence_base = np.array(sequence_base)
    
    print("\nTransformation mathématique des mouvements en cours...")
    sept_variations = generer_variations(sequence_base)
    noms_variations = ["Originale", "Bruitée (Tremblement)", "Zoom In", "Zoom Out", "Décalée", "Rotation", "Étirement X"]

    print(f"\n✅ Génération des fichiers à partir de l'index {compteur_actuel} :")
    for i, seq_aug in enumerate(sept_variations):
        nom_fichier = f"{compteur_actuel}.npy"
        np.save(os.path.join('dataset', MOT_LSF, nom_fichier), seq_aug)
        print(f"  - Création de {nom_fichier} (Version : {noms_variations[i]})")
        compteur_actuel += 1

    print(f"\n🎉 Terminé ! Ton dossier '{MOT_LSF}' contient maintenant {compteur_actuel} fichiers.")
else:
    print("❌ Sauvegarde annulée.")