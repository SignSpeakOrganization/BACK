"""
augmentation.py
---------------
Fonctions de Data Augmentation partagées entre collect_data.py et verify_and_extract.py.
Génère 7 variations mathématiques à partir d'une séquence de 30 frames de keypoints Holistic.
"""

import numpy as np


# ==========================================
# 🔄 TRANSFORMATION MATHÉMATIQUE
# ==========================================
def appliquer_transformation(sequence, scale_x=1.0, scale_y=1.0,
                              shift_x=0.0, shift_y=0.0, angle_deg=0.0):
    """
    Applique une transformation affine (scale, translation, rotation) sur une séquence.
    Fonctionne sur les 1662 keypoints holistic (Pose 132 + Visage+Mains 1530).
    """
    seq_trans = sequence.copy()
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    for f in range(len(seq_trans)):
        if np.all(seq_trans[f] == 0):
            continue  # Ignore les frames vides

        # --- Pose (indices 0..131, format x, y, z, visibility) ---
        for i in range(0, 132, 4):
            nx, ny = seq_trans[f, i] - 0.5, seq_trans[f, i + 1] - 0.5
            rx, ry = nx * c - ny * s, nx * s + ny * c
            seq_trans[f, i]     = (rx * scale_x) + 0.5 + shift_x
            seq_trans[f, i + 1] = (ry * scale_y) + 0.5 + shift_y

        # --- Visage + Mains (indices 132..1661, format x, y, z) ---
        for i in range(132, 1662, 3):
            if seq_trans[f, i] == 0 and seq_trans[f, i + 1] == 0:
                continue
            nx, ny = seq_trans[f, i] - 0.5, seq_trans[f, i + 1] - 0.5
            rx, ry = nx * c - ny * s, nx * s + ny * c
            seq_trans[f, i]     = (rx * scale_x) + 0.5 + shift_x
            seq_trans[f, i + 1] = (ry * scale_y) + 0.5 + shift_y

    return seq_trans


# ==========================================
# 🎲 GÉNÉRATION DES 7 VARIATIONS
# ==========================================
NOMS_VARIATIONS = [
    "Originale",
    "Bruitée (Tremblement)",
    "Zoom In",
    "Zoom Out",
    "Décalée",
    "Rotation",
    "Étirement X",
]


def generer_variations(sequence_de_base):
    """
    Reçoit une séquence numpy (30, 1662) et retourne une liste de 7 variations.
    """
    variations = []

    # 1. Originale
    variations.append(sequence_de_base.copy())

    # 2. Bruit léger (tremblement)
    bruit = np.random.normal(0, 0.003, sequence_de_base.shape)
    seq_bruitee = sequence_de_base + bruit
    seq_bruitee[sequence_de_base == 0] = 0   # Protège les zéros (keypoints absents)
    variations.append(seq_bruitee)

    # 3–7. Transformations géométriques
    variations.append(appliquer_transformation(sequence_de_base, scale_x=1.2,  scale_y=1.2))          # Zoom In
    variations.append(appliquer_transformation(sequence_de_base, scale_x=0.85, scale_y=0.85))         # Zoom Out
    variations.append(appliquer_transformation(sequence_de_base, shift_x=0.06, shift_y=-0.04))        # Décalage
    variations.append(appliquer_transformation(sequence_de_base, angle_deg=10))                       # Rotation
    variations.append(appliquer_transformation(sequence_de_base, scale_x=1.3,  scale_y=1.0))          # Stretch X

    return variations


# ==========================================
# 🎯 DÉTECTION DU MEILLEUR SEGMENT ACTIF
# ==========================================
def trouver_meilleur_segment(sequence_complete, longueur=30):
    """
    Parmi toutes les fenêtres de `longueur` frames dans la séquence complète,
    retourne celle dont le mouvement global est le plus intense.

    Score = somme des normes L2 des différences entre frames consécutives.
    """
    seq = np.array(sequence_complete)
    n = len(seq)

    if n <= longueur:
        # Séquence trop courte → on pad avec des zéros
        padding = np.zeros((longueur - n, seq.shape[1]))
        return np.vstack([seq, padding])

    meilleur_score = -1
    meilleur_debut = 0

    for debut in range(n - longueur + 1):
        fenetre = seq[debut: debut + longueur]
        # Score = énergie cinétique approximée
        diffs = np.diff(fenetre, axis=0)          # (29, 1662)
        score = np.linalg.norm(diffs)
        if score > meilleur_score:
            meilleur_score = score
            meilleur_debut = debut

    return seq[meilleur_debut: meilleur_debut + longueur]


# ==========================================
# 💾 SAUVEGARDE DES VARIATIONS
# ==========================================
def sauvegarder_variations(sequence_base, dossier_destination, compteur_debut):
    """
    Génère les 7 variations et les sauvegarde dans `dossier_destination`.
    Retourne le nouveau compteur après sauvegarde.
    """
    import os
    os.makedirs(dossier_destination, exist_ok=True)

    sept_variations = generer_variations(sequence_base)
    compteur = compteur_debut

    print(f"\n✅ Sauvegarde de 7 variations à partir de l'index {compteur} :")
    for i, seq_aug in enumerate(sept_variations):
        nom_fichier = f"{compteur}.npy"
        np.save(os.path.join(dossier_destination, nom_fichier), seq_aug)
        print(f"  - {nom_fichier}  →  {NOMS_VARIATIONS[i]}")
        compteur += 1

    return compteur
