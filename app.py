#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

from flask import Flask, Response, jsonify
from flask_cors import CORS
from threading import Thread

# --- VARIABLES GLOBALES ---
cap = None
debug_image = None # Pour partager l'image dessinée avec le flux web
current_word = "..." # Remplace hand_sign_letter
process = None

# --- CONFIGURATION DE L'IA ---
ACTIONS = np.array(['neutre', 'bonjour']) # DOIT ÊTRE DANS LE MÊME ORDRE QUE TON NOTEBOOK !
MODEL_PATH_MP = 'hand_landmarker.task'

# On charge le cerveau qu'on vient d'entraîner
try:
    lstm_model = load_model('modele_lsf.keras')
    print("Modèle LSTM chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

# Fonctions de dessin
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

# --- BOUCLE PRINCIPALE (Thread) ---
def main():
    global cap, debug_image, current_word
    print("Lancement de l'analyse vidéo en arrière-plan...")

    # Configuration MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH_MP)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )

    cap = cv.VideoCapture(1) # Met 0 si tu utilises l'iPhone, 1 pour le Mac
    sequence = []
    seuil_confiance = 0.8

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv.flip(frame, 1)
            
            # Préparation pour MediaPipe
            rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Détection des mains
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

            # Gestion de la séquence pour l'IA
            sequence.append(keypoints)
            sequence = sequence[-30:] # On garde les 30 dernières frames

            if len(sequence) == 30:
                # Prédiction de l'IA
                res = lstm_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                if res[np.argmax(res)] > seuil_confiance:
                    current_word = ACTIONS[np.argmax(res)]
                else:
                    current_word = "..."

            # Affichage texte sur la vidéo web
            cv.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            cv.putText(frame, f"Prediction : {current_word}", (15,30), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

            # Mise à jour de l'image globale pour le flux Flask
            debug_image = frame

# --- ROUTES FLASK ---
app = Flask(__name__)
CORS(app)

def generate_frames():
    global debug_image
    while True:
        if debug_image is None:
            continue 
        
        # On encode l'image traitée (avec les points et le texte) pour le navigateur
        ret, buffer = cv.imencode('.jpg', debug_image)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/start', methods=['GET'])
def start():
    global process
    if process is None or not process.is_alive():
        process = Thread(target=main)
        process.start()
        return jsonify({"status": "IA démarrée"}), 200
    else:
        return jsonify({"status": "IA déjà en cours"}), 400

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sign', methods=['GET'])
def sign():
    global process, current_word
    if process is not None and process.is_alive():
        return jsonify({"hand_sign": current_word}), 200
    else:
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