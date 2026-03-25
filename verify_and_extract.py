import cv2
import numpy as np
import mediapipe as mp
import os
from augmentation import trouver_meilleur_segment, sauvegarder_variations

# ==========================================
# ⚙️  CONFIGURATION
# ==========================================
LONGUEUR_SEQUENCE = 30     # Nombre de frames par séquence

# Initialisation de MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils


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
# 🎬 TRAITEMENT D'UNE SEULE VIDÉO
# ==========================================
def traiter_video(video_path, mot_lsf, compteur_actuel):
    """
    Charge la vidéo, affiche le squelette pour vérification, détecte le meilleur
    segment de 30 frames (le plus actif), génère 7 variations et les sauvegarde.
    Retourne le nouveau compteur.
    """
    print(f"\n{'='*55}")
    print(f"  📹 Vidéo : {os.path.basename(video_path)}")
    print(f"{'='*55}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Impossible d'ouvrir : {video_path}")
        return compteur_actuel

    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    delay = max(1, int(1000 / fps))

    sequence_complete = []

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            keypoints = extraire_points_holistic(results)
            sequence_complete.append(keypoints)

            # --- Affichage squelette ---
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(frame, results.pose_landmarks,       mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,  mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Infos à l'écran
            n_frames = len(sequence_complete)
            cv2.putText(frame, f"VERIFICATION : {mot_lsf}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Frame capturee : {n_frames}", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "ESC = ignorer cette video", (15, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)

            cv2.imshow('Verification Video', frame)
            if cv2.waitKey(delay) & 0xFF == 27:   # ESC = on annule cette vidéo
                cap.release()
                cv2.destroyAllWindows()
                print("  ⏭  Vidéo ignorée (ESC).")
                return compteur_actuel

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  📊 {len(sequence_complete)} frames extraites.")

    # --- Détection automatique du meilleur segment ---
    sequence_base = trouver_meilleur_segment(sequence_complete, LONGUEUR_SEQUENCE)
    print(f"  🎯 Meilleur segment de {LONGUEUR_SEQUENCE} frames sélectionné automatiquement.")

    # --- Validation utilisateur ---
    reponse = input("\n  ▶ Les landmarks étaient-ils bien placés ? [o = sauvegarder / n = ignorer] : ")
    if reponse.strip().lower() != 'o':
        print("  ❌ Sauvegarde annulée pour cette vidéo.")
        return compteur_actuel

    dossier = os.path.join('dataset', mot_lsf)
    return sauvegarder_variations(sequence_base, dossier, compteur_actuel)


# ==========================================
# 🚀 POINT D'ENTRÉE PRINCIPAL
# ==========================================
def main():
    print("\n" + "="*55)
    print("  🤟  Extracteur LSF — Mode Multi-Vidéo")
    print("="*55)

    mot_lsf = input("\nNom du signe (ex: bonjour) : ").strip().lower()
    if not mot_lsf:
        print("❌ Nom de signe vide, abandon.")
        return

    # --- Choix du mode ---
    print("\n  Modes disponibles :")
    print("    [1] Traiter une seule vidéo (chemin complet)")
    print("    [2] Traiter TOUTES les vidéos d'un dossier  (ex: videos/bonjour/)")
    mode = input("\nChoix [1/2, Entrée = 2] : ").strip() or "2"

    if mode == "1":
        chemin = input("Chemin de la vidéo : ").strip()
        if not os.path.isfile(chemin):
            print(f"❌ Fichier introuvable : {chemin}")
            return
        videos = [chemin]

    else:  # mode 2 — dossier
        dossier_videos = input(f"Dossier vidéos [Entrée = videos/{mot_lsf}] : ").strip() \
                         or os.path.join("videos", mot_lsf)
        if not os.path.isdir(dossier_videos):
            print(f"❌ Dossier introuvable : {dossier_videos}")
            return

        extensions = ('.mp4', '.webm', '.avi', '.mov', '.mkv')
        videos = sorted([
            os.path.join(dossier_videos, f)
            for f in os.listdir(dossier_videos)
            if f.lower().endswith(extensions)
        ])
        if not videos:
            print(f"❌ Aucune vidéo trouvée dans {dossier_videos}")
            return
        print(f"\n  📂 {len(videos)} vidéo(s) trouvée(s) dans {dossier_videos}")

    # --- Traitement ---
    dossier_dataset = os.path.join('dataset', mot_lsf)
    os.makedirs(dossier_dataset, exist_ok=True)
    compteur = len(os.listdir(dossier_dataset))

    print(f"\n  💾 Dataset actuel : {compteur} fichiers dans dataset/{mot_lsf}/")

    for i, video_path in enumerate(videos):
        print(f"\n[Vidéo {i+1}/{len(videos)}]")
        compteur = traiter_video(video_path, mot_lsf, compteur)

    print(f"\n{'='*55}")
    print(f"  🎉 Terminé ! dataset/{mot_lsf}/ contient maintenant {compteur} fichiers.")
    print(f"  (soit ~{compteur // 7} vidéo(s) source × 7 variations)")
    print("="*55)


if __name__ == '__main__':
    main()