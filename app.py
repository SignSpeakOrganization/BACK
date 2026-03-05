#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import cv2 as cv
import numpy as np
import mediapipe as mp
import copy
import itertools
import csv
from collections import deque
from tensorflow.keras.models import load_model

from flask import Flask, Response, jsonify
from flask_cors import CORS
from threading import Thread

# --- IMPORTS DE TES ANCIENS MODÈLES (LETTRES) ---
from model import KeyPointClassifier

# --- VARIABLES GLOBALES ---
cap = None
debug_image = None
final_prediction = "..."  
current_mode = "..."      
process = None

# --- MEDIAPIPE HOLISTIC ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 🧠 DÉTECTION DYNAMIQUE DES MOTS
# ==========================================
DATA_PATH = os.path.join('dataset')
if os.path.exists(DATA_PATH):
    # On liste les dossiers, et on les trie par ordre alphabétique !
    # Le tri est crucial pour correspondre à l'ordre d'entraînement de ton Jupyter
    ACTIONS = np.array(sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]))
else:
    ACTIONS = np.array(['neutre', 'bonjour']) # Sécurité si le dossier n'existe pas
    
print(f"✅ Mots dynamiques détectés dans le dataset : {ACTIONS}")

try:
    lstm_model = load_model('modele_lsf.keras')
    print("✅ Modèle LSTM (Mots) chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur chargement LSTM : {e}")

# --- CONFIGURATION IA (LETTRES STATIQUES) ---
keypoint_classifier = KeyPointClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

# ==========================================
# 🛠️ FONCTIONS UTILITAIRES
# ==========================================
def extraire_points_holistic(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value > 0 else 0
    return list(map(normalize_, temp_landmark_list))

# ==========================================
# 🚀 BOUCLE PRINCIPALE (IA)
# ==========================================
def main():
    global cap, debug_image, final_prediction, current_mode
    print("Lancement de l'analyse avec le Juge Mathématique (Holistic)...")

    cap = cv.VideoCapture(1) # 1 pour Mac, 0 pour iPhone
    sequence_lstm = []
    wrist_history = deque(maxlen=10)
    seuil_mouvement = 40 
    seuil_confiance_mots = 0.8

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Préparation de l'image
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            
            lettre_devinee = "..."
            mot_devine = "..."
            
            # --- 1. CERVEAU DES LETTRES (Statique) ---
            # On cherche s'il y a une main droite, sinon la main gauche
            main_active = results.right_hand_landmarks if results.right_hand_landmarks else results.left_hand_landmarks
            
            if main_active:
                landmark_list = calc_landmark_list(frame, main_active.landmark)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                lettre_devinee = keypoint_classifier_labels[hand_sign_id]

            # --- 2. CERVEAU DES MOTS (Dynamique) ---
            keypoints_lstm = extraire_points_holistic(results) # 1662 points
            sequence_lstm.append(keypoints_lstm)
            sequence_lstm = sequence_lstm[-30:]
            
            if len(sequence_lstm) == 30:
                res = lstm_model.predict(np.expand_dims(sequence_lstm, axis=0), verbose=0)[0]
                if res[np.argmax(res)] > seuil_confiance_mots:
                    mot_devine = ACTIONS[np.argmax(res)]

            # --- 3. LE JUGE MATHÉMATIQUE (Vitesse de mouvement) ---
            if main_active:
                poignet = main_active.landmark[0]
                wrist_x, wrist_y = int(poignet.x * w), int(poignet.y * h)
                wrist_history.append((wrist_x, wrist_y))

                if len(wrist_history) == 10:
                    dx = max([p[0] for p in wrist_history]) - min([p[0] for p in wrist_history])
                    dy = max([p[1] for p in wrist_history]) - min([p[1] for p in wrist_history])
                    distance_totale = dx + dy

                    if distance_totale > seuil_mouvement:
                        current_mode = "DYNAMIQUE (Mot)"
                        final_prediction = mot_devine
                    else:
                        current_mode = "STATIQUE (Lettre)"
                        final_prediction = lettre_devinee
            else:
                wrist_history.clear()
                final_prediction = "..."
                current_mode = "AUCUNE MAIN"

            # --- AFFICHAGE VISUEL ---
            # Dessin du squelette
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Bandeau de texte
            couleur_fond = (0, 100, 0) if "STATIQUE" in current_mode else (0, 0, 150)
            if current_mode == "AUCUNE MAIN": couleur_fond = (50, 50, 50)

            cv.rectangle(frame, (0, 0), (640, 70), couleur_fond, -1)
            cv.putText(frame, f"Mode : {current_mode}", (15, 25), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame, f"Signe : {final_prediction}", (15, 55), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv.LINE_AA)

            debug_image = frame

# ==========================================
# 🌐 ROUTES FLASK
# ==========================================
app = Flask(__name__)
CORS(app)

def generate_frames():
    global debug_image
    while True:
        if debug_image is None: continue 
        ret, buffer = cv.imencode('.jpg', debug_image)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/start', methods=['GET'])
def start():
    global process
    if process is None or not process.is_alive():
        process = Thread(target=main)
        process.start()
        return jsonify({"status": "IA démarrée"}), 200
    return jsonify({"status": "IA déjà en cours"}), 400

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sign', methods=['GET'])
def sign():
    global process, final_prediction, current_mode
    if process is not None and process.is_alive():
        return jsonify({
            "prediction": final_prediction,
            "mode": current_mode
        }), 200
    return jsonify({"status": "Caméra non démarrée"}), 400

@app.route('/end', methods=['GET'])
def end():
    global cap
    if cap:
        cap.release()
        cv.destroyAllWindows()
    return jsonify({"status": "Caméra coupée"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)