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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

from flask import Flask, Response, jsonify
from flask_cors import CORS
from threading import Thread

# --- IMPORTS DE TES ANCIENS MODÈLES (LETTRES) ---
from model import KeyPointClassifier

# --- VARIABLES GLOBALES ---
cap = None
debug_image = None
final_prediction = "..."  # Le résultat unique choisi par le juge
current_mode = "..."      # STATIQUE ou DYNAMIQUE
process = None

# --- CONFIGURATION IA (MOTS DYNAMIQUES LSTM) ---
ACTIONS = np.array(['neutre', 'bonjour']) 
MODEL_PATH_MP = 'hand_landmarker.task'

try:
    lstm_model = load_model('modele_lsf.keras')
    print("Modèle LSTM (Mots) chargé avec succès !")
except Exception as e:
    print(f"Erreur chargement LSTM : {e}")

# --- CONFIGURATION IA (LETTRES STATIQUES) ---
keypoint_classifier = KeyPointClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]


# --- FONCTIONS UTILITAIRES POUR LE DESSIN ET LES LETTRES ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def draw_landmarks(image_bgr, hand_landmarks_list):
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
        for (x, y) in pts:
            cv.circle(annotated, (x, y), 4, (0, 0, 255), -1)
    return annotated

def calc_landmark_list(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(hand_landmarks):
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


# --- BOUCLE PRINCIPALE ---
def main():
    global cap, debug_image, final_prediction, current_mode
    print("Lancement de l'analyse avec le Juge Mathématique...")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH_MP)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )

    cap = cv.VideoCapture(1) # 1 pour Mac, 0 pour iPhone
    sequence_lstm = []
    wrist_history = deque(maxlen=10) # Mémoire des 10 dernières positions du poignet
    seuil_mouvement = 40 # En pixels (tu peux l'augmenter si c'est trop sensible)
    seuil_confiance_mots = 0.8

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            result = landmarker.detect(mp_image)
            keypoints_lstm = np.zeros(126) 
            
            if result.hand_landmarks:
                frame = draw_landmarks(frame, result.hand_landmarks)

                # --- 1. Calculs des deux cerveaux en arrière-plan ---
                # Cerveau Lettres
                landmark_list = calc_landmark_list(frame, result.hand_landmarks[0])
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                lettre_devinee = keypoint_classifier_labels[hand_sign_id]

                # Cerveau Mots
                all_landmarks = []
                for hand in result.hand_landmarks:
                    for lm in hand:
                        all_landmarks.extend([lm.x, lm.y, lm.z])
                arr = np.array(all_landmarks)
                limit = min(len(arr), 126)
                keypoints_lstm[:limit] = arr[:limit]
                
                sequence_lstm.append(keypoints_lstm)
                sequence_lstm = sequence_lstm[-30:]
                
                mot_devine = "..."
                if len(sequence_lstm) == 30:
                    res = lstm_model.predict(np.expand_dims(sequence_lstm, axis=0), verbose=0)[0]
                    if res[np.argmax(res)] > seuil_confiance_mots:
                        mot_devine = ACTIONS[np.argmax(res)]

                # --- 2. LE JUGE MATHÉMATIQUE (Vitesse de mouvement) ---
                # On récupère les coordonnées X, Y du poignet (Landmark 0) en pixels
                poignet = result.hand_landmarks[0][0]
                wrist_x, wrist_y = int(poignet.x * w), int(poignet.y * h)
                wrist_history.append((wrist_x, wrist_y))

                if len(wrist_history) == 10:
                    # On calcule l'écartement maximum du poignet sur les 10 dernières frames
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
            couleur_fond = (0, 100, 0) if "STATIQUE" in current_mode else (0, 0, 150)
            if current_mode == "AUCUNE MAIN": couleur_fond = (50, 50, 50)

            cv.rectangle(frame, (0, 0), (640, 70), couleur_fond, -1)
            cv.putText(frame, f"Mode : {current_mode}", (15, 25), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame, f"Signe : {final_prediction}", (15, 55), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv.LINE_AA)

            debug_image = frame


# --- ROUTES FLASK ---
app = Flask(__name__)
CORS(app)

def generate_frames():
    global debug_image
    while True:
        if debug_image is None:
            continue 
        ret, buffer = cv.imencode('.jpg', debug_image)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/start', methods=['GET'])
def start():
    global process
    if process is None or not process.is_alive():
        process = Thread(target=main)
        process.start()
        return jsonify({"status": "IA Juge démarrée"}), 200
    return jsonify({"status": "IA déjà en cours"}), 400

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sign', methods=['GET'])
def sign():
    global process, final_prediction, current_mode
    if process is not None and process.is_alive():
        # L'API renvoie maintenant une seule prédiction propre avec son contexte !
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