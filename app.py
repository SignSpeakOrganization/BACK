#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

from dotenv import load_dotenv
load_dotenv()

import copy
import csv
import itertools
import time
from collections import deque
from threading import Thread, Event, Lock

import cv2 as cv
import mediapipe as mp
import numpy as np
import requests
from flask import Flask, Response, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

from model import KeyPointClassifier


# ==========================================
# VARIABLES GLOBALES
# ==========================================
cap = None
debug_image = None

final_prediction = "..."
current_mode = "..."
current_confidence = 0.0

last_speech_text = ""
last_speech_error = None

current_phrase = []
last_added_word = None
last_added_time = 0

conversation_history = []
message_counter = 0

process = None
stop_event = Event()
frame_lock = Lock()

camera_running = False
camera_index = None
last_error = None
last_frame_ok = False


# ==========================================
# CONFIG
# ==========================================
CAMERA_INDEX = int(os.getenv("CAMERA_DEVICE", 0))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", 960))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", 540))
MIN_DETECTION = float(os.getenv("MIN_DETECTION_CONFIDENCE", 0.5))
MIN_TRACKING = float(os.getenv("MIN_TRACKING_CONFIDENCE", 0.5))
STT_URL = os.getenv("STT_URL", "http://127.0.0.1:5010/stt")


# ==========================================
# FLASK
# ==========================================
app = Flask(__name__)
CORS(app)


# ==========================================
# MEDIAPIPE
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ==========================================
# DATASET / MODELE
# ==========================================
DATA_PATH = os.path.join("dataset")
if os.path.exists(DATA_PATH):
    ACTIONS = np.array(
        sorted(
            [
                d for d in os.listdir(DATA_PATH)
                if os.path.isdir(os.path.join(DATA_PATH, d))
            ]
        )
    )
else:
    ACTIONS = np.array(["neutre", "bonjour"])

print(f"✅ Mots dynamiques détectés dans le dataset : {ACTIONS}")

try:
    lstm_model = load_model("modele_lsf.keras")
    print("✅ Modèle LSTM (Mots) chargé avec succès !")

    n_classes_model = lstm_model.output_shape[-1]
    if n_classes_model != len(ACTIONS):
        print(
            f"⚠️  ATTENTION : Le modèle a été entraîné avec {n_classes_model} classes "
            f"mais le dataset en contient {len(ACTIONS)}. "
            f"Réentraîne le modèle pour éviter des erreurs d'index."
        )
except Exception as e:
    print(f"❌ Erreur chargement LSTM : {e}")
    lstm_model = None

keypoint_classifier = KeyPointClassifier()

with open(
    "model/keypoint_classifier/keypoint_classifier_label.csv",
    encoding="utf-8-sig"
) as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]


# ==========================================
# OUTILS
# ==========================================
def extraire_points_holistic(results):
    pose = np.array(
        [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
    ).flatten() if results.pose_landmarks else np.zeros(132)

    face = np.array(
        [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
    ).flatten() if results.face_landmarks else np.zeros(1404)

    lh = np.array(
        [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(63)

    rh = np.array(
        [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([pose, face, lh, rh])


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks:
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


def add_to_history(source, text):
    global conversation_history, message_counter

    if text is None:
        return

    text = text.strip()
    if text == "" or text == "..." or text == "analyse ...":
        return

    message_counter += 1
    conversation_history.append({
        "id": message_counter,
        "timestamp": time.time(),
        "source": source,
        "text": text
    })


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
    add_to_history("lsf", word)


def call_stt_api(audio_path, lang="fr"):
    global last_speech_error

    try:
        with open(audio_path, "rb") as audio_file:
            files = {"audio": audio_file}
            data = {"lang": lang}

            response = requests.post(STT_URL, files=files, data=data, timeout=120)

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
# BOUCLE PRINCIPALE
# ==========================================
def main():
    global cap, debug_image, final_prediction, current_mode
    global current_confidence, camera_running, camera_index
    global last_error, last_frame_ok

    print("Lancement de l'analyse Holistic...")
    stop_event.clear()

    camera_running = True
    camera_index = CAMERA_INDEX
    last_error = None
    last_frame_ok = False

    final_prediction = "..."
    current_mode = "..."
    current_confidence = 0.0

    cap = cv.VideoCapture(CAMERA_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        last_error = f"Impossible d'ouvrir la caméra index {CAMERA_INDEX}"
        camera_running = False
        print(f"[ERROR] {last_error}")
        return

    sequence_lstm = []
    wrist_history = deque(maxlen=10)
    seuil_mouvement = int(0.06 * CAMERA_WIDTH)
    seuil_confiance_mots = 0.8

    with mp_holistic.Holistic(
        min_detection_confidence=MIN_DETECTION,
        min_tracking_confidence=MIN_TRACKING
    ) as holistic:

        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()

            if not ret or frame is None:
                last_frame_ok = False
                last_error = "Lecture frame échouée"
                time.sleep(0.05)
                continue

            last_frame_ok = True
            last_error = None

            frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]

            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            lettre_devinee = "..."
            mot_devine = "..."
            current_confidence = 0.0

            main_active = (
                results.right_hand_landmarks
                if results.right_hand_landmarks
                else results.left_hand_landmarks
            )

            if main_active:
                landmark_list = calc_landmark_list(frame, main_active.landmark)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                    lettre_devinee = keypoint_classifier_labels[hand_sign_id]

            keypoints_lstm = extraire_points_holistic(results)
            sequence_lstm.append(keypoints_lstm)
            sequence_lstm = sequence_lstm[-30:]

            if len(sequence_lstm) == 30 and lstm_model is not None:
                try:
                    res = lstm_model.predict(
                        np.expand_dims(sequence_lstm, axis=0),
                        verbose=0
                    )[0]
                    best_idx = int(np.argmax(res))
                    current_confidence = float(res[best_idx])

                    if best_idx < len(ACTIONS) and current_confidence > seuil_confiance_mots:
                        mot_devine = ACTIONS[best_idx]
                except Exception as e:
                    last_error = f"Erreur prédiction LSTM: {str(e)}"

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
                current_confidence = 0.0

            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
            )
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            couleur_fond = (0, 100, 0) if "STATIQUE" in current_mode else (0, 0, 150)
            if current_mode == "AUCUNE MAIN":
                couleur_fond = (50, 50, 50)

            cv.rectangle(frame, (0, 0), (640, 70), couleur_fond, -1)
            cv.putText(
                frame,
                f"Mode : {current_mode}",
                (15, 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                f"Signe : {final_prediction}",
                (15, 55),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv.LINE_AA,
            )

            with frame_lock:
                debug_image = frame.copy()

    if cap is not None:
        cap.release()
        cap = None

    camera_running = False
    print("[DEBUG] Fin du thread caméra")


# ==========================================
# ROUTES
# ==========================================
def generate_frames():
    global debug_image

    while True:
        with frame_lock:
            if debug_image is None:
                time.sleep(0.03)
                continue
            frame_copy = debug_image.copy()

        ret, buffer = cv.imencode(".jpg", frame_copy)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "API LSF active",
        "dataset": list(ACTIONS),
        "endpoints": {
            "GET /health": "Etat du backend",
            "GET /start": "Allume la caméra et lance la détection",
            "GET /status": "Etat caméra et thread",
            "GET /sign": "Retourne la prédiction courante",
            "GET /phrase": "Retourne la phrase courante",
            "GET /speech/test": "Teste la transcription vocale",
            "GET /conversation/state": "Etat global",
            "GET /video_feed": "Flux vidéo MJPEG",
            "GET /end": "Coupe la caméra"
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "service": "SignSpeak BACK",
        "status": "ok",
        "timestamp": time.time()
    }), 200


@app.route("/start", methods=["GET"])
def start():
    global process

    if process is None or not process.is_alive():
        process = Thread(target=main, daemon=True)
        process.start()
        return jsonify({
            "success": True,
            "status": "IA démarrée",
            "camera_running": True
        }), 200

    return jsonify({
        "success": False,
        "status": "IA déjà en cours",
        "camera_running": True
    }), 400


@app.route("/status", methods=["GET"])
def status():
    global process, camera_running, camera_index
    global final_prediction, current_mode, current_confidence
    global last_frame_ok, last_error, current_phrase

    return jsonify({
        "success": True,
        "thread_alive": process.is_alive() if process is not None else False,
        "camera_running": camera_running,
        "camera_index": camera_index,
        "frame_ok": last_frame_ok,
        "prediction": final_prediction,
        "mode": current_mode,
        "confidence": current_confidence,
        "phrase_length": len(current_phrase),
        "text": " ".join(current_phrase),
        "error": last_error,
        "timestamp": time.time()
    }), 200


@app.route("/sign", methods=["GET"])
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


@app.route("/phrase", methods=["GET"])
def phrase():
    return jsonify({
        "phrase": current_phrase,
        "text": " ".join(current_phrase)
    }), 200


@app.route("/clear_phrase", methods=["GET", "POST"])
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


@app.route("/speech/test", methods=["GET"])
def speech_test():
    global last_speech_text, last_speech_error

    test_audio_path = "test.wav"

    result = call_stt_api(test_audio_path, lang="fr")

    if result.get("status") == "ok":
        last_speech_text = result.get("text", "")
        add_to_history("speech", last_speech_text)

        return jsonify({
            "success": True,
            "speech_text": last_speech_text,
            "language": result.get("language"),
            "segments": result.get("segments", []),
            "error": None
        }), 200

    return jsonify({
        "success": False,
        "speech_text": last_speech_text,
        "error": result.get("error", "Erreur STT inconnue"),
        "speech_error": last_speech_error
    }), 500
@app.route("/conversation/history", methods=["GET"])
def get_conversation_history():
    global conversation_history

    return jsonify({
        "success": True,
        "count": len(conversation_history),
        "history": conversation_history
    }), 200
    
@app.route("/conversation/history/clear", methods=["GET", "POST"])
def clear_conversation_history():
    global conversation_history, message_counter

    conversation_history = []
    message_counter = 0

    return jsonify({
        "success": True,
        "status": "Historique effacé",
        "count": 0,
        "history": []
    }), 200

@app.route("/conversation/state", methods=["GET"])
def conversation_state():
    global process, camera_running, camera_index
    global final_prediction, current_mode, current_confidence
    global last_frame_ok, last_error, current_phrase
    global last_speech_text, last_speech_error
    global conversation_history

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
        "speech_text": last_speech_text,
        "speech_error": last_speech_error,
        "history_count": len(conversation_history),
        "last_history_item": conversation_history[-1] if conversation_history else None,
        "error": last_error,
        "timestamp": time.time()
    }), 200


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/end", methods=["GET"])
def end():
    global cap, process, camera_running

    stop_event.set()

    if process is not None:
        process.join(timeout=3)

    if cap is not None:
        cap.release()
        cap = None

    camera_running = False
    cv.destroyAllWindows()

    return jsonify({
        "success": True,
        "status": "Caméra coupée",
        "camera_running": False
    }), 200


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True, use_reloader=False)