#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

# --- CHARGEMENT DU .env ---
from dotenv import load_dotenv
load_dotenv()

import cv2 as cv
import numpy as np
import mediapipe as mp
import copy
import itertools
import csv
import time
import requests
from collections import deque
from threading import Thread, Event, Lock
from tensorflow.keras.models import load_model

from flask import Flask, Response, jsonify
from flask_cors import CORS

# --- IMPORTS DE TES ANCIENS MODÈLES (LETTRES) ---
from model import KeyPointClassifier

# --- VARIABLES GLOBALES ---
cap = None
debug_image = None
final_prediction = "..."  
current_mode = "..."     
current_confidence = 0.0
last_speech_text = ""
last_speech_error = None

# pour import time 
current_phrase = []
last_added_word = None
last_added_time = 0

 
process = None
stop_event = Event()       # Signal d'arrêt propre pour le thread
frame_lock = Lock()        # Protège debug_image contre les race conditions

# --- CONFIG CAMÉRA DEPUIS .env ---
CAMERA_INDEX  = int(os.getenv("CAMERA_DEVICE", 1))
CAMERA_WIDTH  = int(os.getenv("CAMERA_WIDTH", 960))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", 540))
MIN_DETECTION = float(os.getenv("MIN_DETECTION_CONFIDENCE", 0.5))
MIN_TRACKING  = float(os.getenv("MIN_TRACKING_CONFIDENCE", 0.5))

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
    # Vérification de cohérence modèle / dataset
    n_classes_model = lstm_model.output_shape[-1]
    if n_classes_model != len(ACTIONS):
        print(f"⚠️  ATTENTION : Le modèle a été entraîné avec {n_classes_model} classes "
              f"mais le dataset en contient {len(ACTIONS)}. "
              f"Réentraîne le modèle pour éviter des erreurs d'index.")
except Exception as e:
    print(f"❌ Erreur chargement LSTM : {e}")
    lstm_model = None

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

 
def add_word_to_phrase(word):
    global current_phrase, last_added_word, last_added_time

    if word is None or word.strip() == "" or word == "..." or word == "analyse ...":
        return

    now = time.time()

    if word == last_added_word and (now - last_added_time) < 1.5:
        return

    current_phrase.append(word)
    last_added_word = word
    last_added_time = now
    
def call_stt_api(audio_path, lang="fr"):
    global last_speech_error

    url = "http://127.0.0.1:5010/stt"

    try:
        with open(audio_path, "rb") as audio_file:
            files = {
                "audio": audio_file
            }
            data = {
                "lang": lang
            }

            response = requests.post(url, files=files, data=data, timeout=120)
            
            try:
                result = response.json()
            except Exception:
                result = {
                    "status": "error",
                    "error": response.text
                }

            if response.status_code >= 400:
                last_speech_error = result.get("error", f"HTTP {response.status_code}")
                return result

            last_speech_error = None
            return result

    except Exception as e:
        last_speech_error = str(e)
        return {
            "status": "error",
            "error": str(e)
        }
 
# ==========================================
# 🚀 BOUCLE PRINCIPALE (IA)
# ==========================================
def main():
    global cap, debug_image, final_prediction, current_mode
    global current_confidence, camera_running, camera_index
    global last_error, last_frame_ok
    
    print("Lancement de l'analyse avec le Juge Mathématique (Holistic)...")
    stop_event.clear()

    cap = cv.VideoCapture(CAMERA_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    sequence_lstm = []
    wrist_history = deque(maxlen=10)
    # Seuil adaptatif : ~6% de la largeur de l'image (indépendant de la résolution)
    seuil_mouvement = int(0.06 * CAMERA_WIDTH)
    seuil_confiance_mots = 0.8

    with mp_holistic.Holistic(min_detection_confidence=MIN_DETECTION, min_tracking_confidence=MIN_TRACKING) as holistic:
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                last_frame_ok = False
                last_error = "Lecture frame échouée"
                print("[Error] Impossible de lire une frame")
                continue
            last_frame_ok = True
            last_error = None
            
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
            
            if len(sequence_lstm) == 30 and lstm_model is not None:
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
                        final_prediction = mot_devine if mot_devine != "..." else "analyse ..."
                        add_word_to_phrase(final_prediction)
                    else:
                        current_mode = "STATIQUE (Lettre)"
                        final_prediction = lettre_devinee if lettre_devinee != "..." else "analyse ..."
                        
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

            with frame_lock:
                debug_image = frame.copy()

# ==========================================
# 🌐 ROUTES FLASK
# ==========================================
app = Flask(__name__)
CORS(app)

def generate_frames():
    global debug_image
    while True:
        with frame_lock:
            if debug_image is None:
                continue
            frame_copy = debug_image.copy()
        ret, buffer = cv.imencode('.jpg', frame_copy)
        if not ret:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "API LSF active",
        "dataset": list(ACTIONS),
        "endpoints": {
            "GET /start":      "Allume la caméra et lance la détection",
            "GET /sign":       "Retourne la prédiction courante (JSON)",
            "GET /video_feed": "Flux MJPEG de la caméra avec squelette",
            "GET /end":        "Coupe la caméra et arrête le thread"
        }
    }), 200

@app.route('/start', methods=['GET'])
def start():
    global process
    if process is None or not process.is_alive():
        process = Thread(target=main, daemon=True)
        process.start()
        return jsonify({"status": "IA démarrée"}), 200
    return jsonify({"status": "IA déjà en cours"}), 400

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/conversation/state', methods=['GET'])
def conversation_state():
    global process, camera_running, camera_index
    global final_prediction, current_mode, current_confidence
    global last_frame_ok, last_error, current_phrase
    global last_speech_text, last_speech_error

    return jsonify({
        "success": True,
        "thread_alive": process.is_alive() if process is not None else False,
        "camera_running": camera_running,
        "camera_index": camera_index,
        "frame_ok": last_frame_ok,
        "prediction": final_prediction,
        "mode": current_mode,
        "confidence": current_confidence,
        "phrase": current_phrase,
        "text": " ".join(current_phrase),
        "error": last_error,
        "speech_text": last_speech_text,
        "speech_error": last_speech_error,
        "timestamp": time.time()
    }), 200

@app.route('/sign', methods=['GET'])
def sign():
    global process, final_prediction, current_mode
    global current_confidence, camera_running, last_frame_ok, last_error

    if process is not None and process.is_alive():
        return jsonify({
            "success": True,
            "prediction": final_prediction,
            "mode": current_mode,
            "confidence": current_confidence,
            "camera_running": camera_running,
            "frame_ok": last_frame_ok,
            "error": last_error,
            "timestamp": time.time()
        }), 200

    return jsonify({
        "success": False,
        "prediction": "",
        "mode": "",
        "confidence": 0.0,
        "camera_running": False,
        "frame_ok": False,
        "error": "Caméra non démarrée",
        "timestamp": time.time()
    }), 400
    
@app.route('/phrase', methods=['GET'])
def phrase():
    global current_phrase
    return jsonify({
        "phrase": current_phrase,
        "text": " ".join(current_phrase)
    }), 200
    
@app.route('/clear_phrase', methods=['POST', 'GET'])
def clear_phrase():
    global current_phrase, last_added_word, last_added_time

    current_phrase = []
    last_added_word = None
    last_added_time = 0

    return jsonify({
        "status": "Phrase effacée",
        "phrase": current_phrase,
        "text": ""
    }), 200

@app.route('/end', methods=['GET'])
def end():
    global cap, process
    stop_event.set()           # Demande l'arrêt propre du thread
    if process is not None:
        process.join(timeout=3)  # Attend max 3s que le thread se termine
    if cap is not None:
        cap.release()
        cap = None
    cv.destroyAllWindows()
    return jsonify({"status": "Caméra coupée"}), 200

if __name__ == "__main__":
    flask_host  = os.getenv("FLASK_HOST",  "0.0.0.0")
    flask_port  = int(os.getenv("FLASK_PORT", 5000))
    flask_debug = os.getenv("FLASK_DEBUG", "0") == "1"
    # use_reloader=False : indispensable quand on utilise des Threads,
    # le reloader démarrerait 2 processus et causerait un conflit sur la caméra.
    app.run(host=flask_host, port=flask_port, debug=flask_debug, use_reloader=False)