# SignSpeak — Cadrage fonctionnel : traduction bidirectionnelle Voix ↔ Texte

## 1. Contexte et besoin
SignSpeak vise à améliorer l’accessibilité des visioconférences en facilitant la communication entre personnes entendantes et personnes sourdes/malentendantes.  
Dans ce cadre, la traduction **Voix ↔ Texte** complète la traduction LSF ↔ Texte : elle permet d’afficher ce que disent les personnes entendantes et de vocaliser des messages textuels pour fluidifier les échanges.

## 2. Objectifs
### Objectif principal (MVP)
Mettre à disposition une fonctionnalité bidirectionnelle :
- **Voix → Texte (STT)** : transcrire la parole en texte affiché dans l’interface.
- **Texte → Voix (TTS)** : vocaliser un texte (saisi ou généré) via une synthèse vocale.

### Objectifs secondaires
- Fournir un retour utilisateur clair (état d’écoute, transcription en cours, erreurs).
- Garantir une latence acceptable et une utilisation simple (1 clic).
- Préparer l’intégration future dans l’extension et l’historique de conversation.

## 3. Périmètre
### Inclus (MVP)
- Transcription **FR** (fr-FR) par segments (push-to-talk ou enregistrement court).
- Affichage du texte transcrit dans une bulle/zone de chat.
- Lecture vocale d’un texte via une voix FR.
- Gestion d’erreurs minimale (micro refusé, audio invalide, service indisponible).

### Exclu (hors MVP)
- Traduction automatique multilingue.
- Reconnaissance multi-locuteurs (diarisation) / identification des intervenants.
- Traduction en streaming “mot à mot” parfaitement temps réel.
- Ponctuation parfaite, résumé automatique, correction grammaticale avancée.

## 4. Utilisateurs et scénarios
### Persona 1 : collaborateur entendant
- Parle pendant une réunion.
- Le système transcrit et affiche son message en texte.

### Persona 2 : utilisateur qui communique via texte
- Tape un message ou sélectionne un texte généré (ex : phrase issue des signes).
- Le système lit le texte à voix haute.

### Scénarios clés
1) **STT — push-to-talk**
- L’utilisateur clique “🎙️ Parler”.
- Le micro enregistre 3–10 secondes.
- Le texte apparaît et est enregistré dans l’historique.

2) **TTS — lecture**
- L’utilisateur clique “🔊 Lire”.
- Le texte est vocalisé (sortie audio locale).

## 5. Règles fonctionnelles
### Voix → Texte (STT)
- L’utilisateur doit pouvoir :
  - démarrer/arrêter l’enregistrement,
  - voir un indicateur “écoute en cours”,
  - obtenir une transcription ou un message d’erreur.
- Le texte final est affiché et horodaté.
- Une option simple “Annuler” supprime la dernière transcription si besoin.

### Texte → Voix (TTS)
- L’utilisateur peut :
  - sélectionner un texte (zone de saisie ou bulle),
  - lancer la lecture,
  - arrêter la lecture.
- La voix utilisée est une voix française disponible (choix ultérieur possible).

## 6. Interfaces attendues (niveau UX)
### Composants
- Bouton 🎙️ **Parler** (états : prêt / enregistrement / traitement)
- Zone d’affichage texte (bulle ou chat)
- Bouton 🔊 **Lire** (lecture / stop)
- Messages système (micro refusé, pas de son, erreur serveur)

### Feedback minimal requis
- “Enregistrement…”  
- “Transcription…”  
- “Texte transcrit : …”  
- “Erreur : micro non autorisé / audio non valide / service indisponible”

## 7. Données et historique
Format minimal d’un message voix/texte (pour stockage futur) :
- `id` (uuid)
- `type` : `voice_to_text` ou `text_to_voice`
- `content` : texte
- `timestamp`
- `source` (optionnel) : `user` / `system`

## 8. Performance & qualité attendues
- **Latence cible (MVP)** : 1 à 3 secondes après fin d’enregistrement (variable selon solution).
- **Robustesse** : fonctionne avec un micro standard, supporte bruit modéré.
- **Accessibilité** : boutons lisibles, labels clairs, navigation simple.

## 9. Dépendances et contraintes
- Autorisation micro côté navigateur / OS.
- Qualité du résultat dépend :
  - du modèle/service STT,
  - du bruit,
  - de l’accent/débit de parole.
- RGPD : si audio envoyé à un service externe, le préciser (à valider).

## 10. Critères d’acceptation (Definition of Done)
### STT
- Un utilisateur peut enregistrer sa voix et obtenir un texte affiché.
- En cas de refus micro, un message clair apparaît.
- Le texte est récupérable via l’interface (et prêt pour l’historique).

### TTS
- Un utilisateur peut saisir ou sélectionner un texte et le faire lire.
- Le bouton stop interrompt la lecture.

## 11. Risques & points à valider
- Choix technique final (local vs API cloud) selon contraintes RGPD et performances.
- Support navigateur (Chrome prioritaire pour extension).
- Gestion multi-onglets/plusieurs sessions.

## 12. Prochaines étapes (après validation cadrage)
1) Prototyper STT (push-to-talk) et TTS (lecture).
2) Définir routes API et payloads (si backend requis).
3) Intégrer à l’interface extension + ajout à l’historique.
4) Tests utilisateur (5 personnes, 10 phrases, bruit modéré).