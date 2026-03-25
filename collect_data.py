import cv2
import numpy as np
import mediapipe as mp
import os
import time
from augmentation import sauvegarder_variations

# ==========================================
# ⚙️  CONFIGURATION
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

LONGUEUR_SEQUENCE = 30   # frames par essai


# ==========================================
# 🧠 EXTRACTION DES 1662 POINTS
# ==========================================
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


# ==========================================
# 🖼️  DESSIN DU SQUELETTE
# ==========================================
def dessiner_squelette(frame, results):
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(frame, results.pose_landmarks,       mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,  mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# ==========================================
# ⏳  PHASE COMPTE-À-REBOURS (preview live)
# ==========================================
def phase_countdown(cap, holistic, sequence_num, nb_sequences, action, duree_secondes=3):
    """
    Affiche un live-preview avec un bandeau jaune + décompte.
    Les frames sont LUES et affichées (le user se voit en temps réel),
    mais NON sauvegardées. Retourne True si on continue, False si ESC enfoncé.
    """
    debut = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            return False
        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        dessiner_squelette(frame, results)

        restant = duree_secondes - (time.time() - debut)
        if restant <= 0:
            return True

        # Bandeau jaune = Prépare-toi
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (0, 180, 220), -1)
        cv2.putText(frame, f"Essai {sequence_num} / {nb_sequences}  |  {action.upper()}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"PREPARE-TOI ... {int(restant) + 1}",
                    (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 80), 2, cv2.LINE_AA)

        cv2.imshow('Collecte LSF Holistic', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return False   # ESC = arrêt global


# ==========================================
# 🔴  PHASE ENREGISTREMENT (30 frames)
# ==========================================
def phase_enregistrement(cap, holistic, sequence_num, nb_sequences, action):
    """
    Capture exactement LONGUEUR_SEQUENCE frames, affiche un indicateur REC rouge
    avec le numéro de frame en cours. Retourne le tableau numpy (30, 1662) ou None si ESC.
    """
    window = []

    for frame_num in range(LONGUEUR_SEQUENCE):
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Extraction & stockage
        keypoints = extraire_points_holistic(results)
        window.append(keypoints)

        dessiner_squelette(frame, results)

        # --- Bandeau rouge = ENREGISTREMENT ---
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (0, 0, 180), -1)

        # Indicateur "● REC" clignotant (visible sur les frames paires)
        if frame_num % 6 < 3:
            cv2.putText(frame, "● REC", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 60, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Essai {sequence_num} / {nb_sequences}  |  {action.upper()}",
                    (110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

        # Barre de progression
        progression = int((frame_num + 1) / LONGUEUR_SEQUENCE * (frame.shape[1] - 30))
        cv2.rectangle(frame, (15, 48), (15 + progression, 65), (0, 255, 100), -1)
        cv2.rectangle(frame, (15, 48), (frame.shape[1] - 15, 65), (255, 255, 255), 1)  # contour
        cv2.putText(frame, f"Frame {frame_num + 1:02d}/{LONGUEUR_SEQUENCE}",
                    (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('Collecte LSF Holistic', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return None   # ESC = arrêt global

    return np.array(window)   # shape (30, 1662)


# ==========================================
# 🚀 POINT D'ENTRÉE PRINCIPAL
# ==========================================
def main():
    print("\n" + "="*55)
    print("  🤟  Collecte LSF — Webcam")
    print("="*55)

    action = input("\nNom du signe (ex: bonjour) : ").strip().lower()
    if not action:
        print("❌ Nom vide, abandon.")
        return

    try:
        nb_sequences = int(input("Combien d'essais ? [Entrée = 30] : ") or 30)
        cam_index    = int(input("Caméra [0 = iPhone/externe, 1 = Mac intégrée, Entrée = 1] : ") or 1)
        augmenter    = input("Appliquer data augmentation ×7 ? [o/n, Entrée = o] : ").strip().lower() or "o"
    except ValueError:
        nb_sequences = 30
        cam_index    = 1
        augmenter    = "o"

    dossier_action = os.path.join('dataset', action)
    os.makedirs(dossier_action, exist_ok=True)
    compteur_actuel = len(os.listdir(dossier_action))

    print(f"\n  📁 Dataset actuel : {compteur_actuel} fichiers dans dataset/{action}/")
    print(f"  📹 Caméra {cam_index}  |  {nb_sequences} essais  |  augmentation = {'OUI ×7' if augmenter == 'o' else 'NON'}")
    print(f"\n  💡 JAUNE = préparation  |  ROUGE = enregistrement en cours")
    print(f"  [ESC à tout moment pour arrêter proprement]\n")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la caméra {cam_index}")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:

        for sequence in range(1, nb_sequences + 1):

            # --- Compte-à-rebours (preview live, pas de sauvegarde) ---
            if not phase_countdown(cap, holistic, sequence, nb_sequences, action):
                print("\n⏹  Arrêt demandé (ESC).")
                break

            # --- Enregistrement des 30 frames ---
            window = phase_enregistrement(cap, holistic, sequence, nb_sequences, action)
            if window is None:
                print("\n⏹  Arrêt demandé (ESC).")
                break

            # --- Sauvegarde ---
            if augmenter == "o":
                compteur_actuel = sauvegarder_variations(window, dossier_action, compteur_actuel)
                print(f"  ✅ Essai {sequence} → 7 variations sauvegardées")
            else:
                npy_path = os.path.join(dossier_action, str(compteur_actuel))
                np.save(npy_path, window)
                print(f"  ✅ Essai {sequence} → sauvegardé sous {compteur_actuel}.npy")
                compteur_actuel += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🎉 Terminé ! dataset/{action}/ contient maintenant {compteur_actuel} fichiers.")


if __name__ == '__main__':
    main()