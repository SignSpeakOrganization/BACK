import cv2
import numpy as np
import mediapipe as mp
import os
import csv
import copy
import itertools
from tensorflow.keras.models import load_model
from collections import deque
from model import KeyPointClassifier

# ==========================================
# ⚙️  CONFIGURATION
# ==========================================
DATA_PATH            = 'dataset'
SEUIL_CONFIANCE_MOTS = 0.7    # Seuil LSTM (mots dynamiques)
SEUIL_MOUVEMENT      = 0.06   # ~6% de la largeur image
CAM_INDEX            = int(os.getenv("CAMERA_DEVICE", 1))


# ==========================================
# 🧠 MODÈLE LSTM — MOTS DYNAMIQUES
# ==========================================
if os.path.exists(DATA_PATH):
    ACTIONS = np.array(sorted([
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    ]))
else:
    raise FileNotFoundError("Dossier 'dataset/' introuvable.")

lstm_model = load_model('modele_lsf.keras')
n_classes_model = lstm_model.output_shape[-1]
print(f"✅ Modèle LSTM chargé  |  Classes : {list(ACTIONS)}")
if n_classes_model != len(ACTIONS):
    print(f"⚠️  Modèle entraîné avec {n_classes_model} classes, dataset en a {len(ACTIONS)} — réentraîne !")


# ==========================================
# ✋ MODÈLE TFLite — LETTRES STATIQUES
# ==========================================
keypoint_classifier = KeyPointClassifier()
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
print(f"✅ Classifier statique chargé  |  Lettres : {keypoint_classifier_labels}")


# ==========================================
# 🛠️  FONCTIONS UTILITAIRES
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

def extraire_points_holistic(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(132)
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() \
           if results.face_landmarks else np.zeros(1404)
    lh   = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(63)
    rh   = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
           if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

def calc_landmark_list(image, landmarks):
    iw, ih = image.shape[1], image.shape[0]
    return [[min(int(lm.x * iw), iw - 1), min(int(lm.y * ih), ih - 1)]
            for lm in landmarks]

def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    bx, by = temp[0]
    temp = [[p[0] - bx, p[1] - by] for p in temp]
    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, flat)) or 1
    return [v / max_val for v in flat]


# ==========================================
# 🚀 BOUCLE PRINCIPALE
# ==========================================
print(f"\nCaméra {CAM_INDEX}  |  ESC = quitter  |  R = reset\n")

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Impossible d'ouvrir la caméra {CAM_INDEX}")

sequence             = []
wrist_history        = deque(maxlen=10)
historique_preds     = deque(maxlen=5)

prediction_affichee  = "..."
mode_actuel          = "AUCUNE MAIN"
confiance_pct        = 0.0
lettre_devinee       = "..."
mot_devine           = "..."

with mp_holistic.Holistic(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # --- Squelette ---
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
        mp_drawing.draw_landmarks(frame, results.pose_landmarks,       mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,  mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        main_active = results.right_hand_landmarks or results.left_hand_landmarks

        # ── 1. LETTRES STATIQUES ───────────────────────────────────────
        lettre_devinee = "..."
        if main_active:
            lm_list = calc_landmark_list(frame, main_active.landmark)
            pre_lm  = pre_process_landmark(lm_list)
            idx_lettre = keypoint_classifier(pre_lm)
            lettre_devinee = keypoint_classifier_labels[idx_lettre]

        # ── 2. MOTS DYNAMIQUES (LSTM) ──────────────────────────────────
        keypoints = extraire_points_holistic(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        mot_devine    = "..."
        confiance_pct = 0.0
        if len(sequence) == 30:
            res = lstm_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            idx_mot = np.argmax(res)
            confiance_pct = float(res[idx_mot]) * 100
            if res[idx_mot] > SEUIL_CONFIANCE_MOTS:
                historique_preds.append(ACTIONS[idx_mot])
                if (len(historique_preds) >= 3 and
                        historique_preds.count(historique_preds[-1]) >= 3):
                    mot_devine = historique_preds[-1]

        # ── 3. JUGE DE MOUVEMENT ───────────────────────────────────────
        if main_active:
            poignet = main_active.landmark[0]
            wrist_history.append((int(poignet.x * w), int(poignet.y * h)))
            if len(wrist_history) == 10:
                dx = max(p[0] for p in wrist_history) - min(p[0] for p in wrist_history)
                dy = max(p[1] for p in wrist_history) - min(p[1] for p in wrist_history)
                if (dx + dy) > int(SEUIL_MOUVEMENT * w):
                    mode_actuel = "DYNAMIQUE"
                    prediction_affichee = mot_devine
                else:
                    mode_actuel = "STATIQUE"
                    prediction_affichee = lettre_devinee
            # Moins de 10 points encore : on attend
        else:
            wrist_history.clear()
            historique_preds.clear()
            mode_actuel         = "AUCUNE MAIN"
            prediction_affichee = "..."
            confiance_pct       = 0.0

        # ==========================================
        # 🖼️  AFFICHAGE
        # ==========================================
        # Bandeau supérieur
        couleurs = {"DYNAMIQUE": (0, 0, 160), "STATIQUE": (0, 110, 0), "AUCUNE MAIN": (50, 50, 50)}
        couleur  = couleurs.get(mode_actuel, (50, 50, 50))
        cv2.rectangle(frame, (0, 0), (w, 85), couleur, -1)

        # Mode
        cv2.putText(frame, f"Mode : {mode_actuel}", (15, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        # Prédiction principale (grande)
        cv2.putText(frame, prediction_affichee, (15, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

        # Prédictions secondaires (petites, en haut à droite)
        cv2.putText(frame, f"Lettre : {lettre_devinee}", (w - 200, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Mot    : {mot_devine}", (w - 200, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)

        # Barre de confiance LSTM (en bas)
        if confiance_pct > 0 and mode_actuel == "DYNAMIQUE":
            barre_w = int((w - 30) * confiance_pct / 100)
            cv2.rectangle(frame, (15, h - 22), (w - 15, h - 8), (60, 60, 60), -1)
            col_barre = (0, 255, 100) if confiance_pct >= 70 else (0, 200, 255)
            cv2.rectangle(frame, (15, h - 22), (15 + barre_w, h - 8), col_barre, -1)
            cv2.putText(frame, f"Confiance LSTM : {confiance_pct:.1f}%", (15, h - 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Touches
        cv2.putText(frame, "ESC = quitter  |  R = reset", (w - 250, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

        cv2.imshow('Test IA LSF — Temps Réel', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        elif key in (ord('r'), ord('R')):
            prediction_affichee = "..."
            confiance_pct       = 0.0
            historique_preds.clear()
            sequence.clear()
            print("🔄 Reset")

cap.release()
cv2.destroyAllWindows()
print("\n👋 Fermeture du test.")