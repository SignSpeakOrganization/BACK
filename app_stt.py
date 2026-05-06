import os
import uuid
import subprocess
import traceback
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel


# -----------------------------
# Config
# -----------------------------
APP_PORT = int(os.getenv("STT_PORT", "5010"))
TMP_DIR = Path(__file__).parent / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Modèle whisper: "small" = bon compromis sur Mac (qualité/latence)
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # cpu sur Mac
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # rapide en CPU

print("[INFO] Chargement du modèle Whisper...")
whisper_model = WhisperModel(
    WHISPER_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)
print("[OK] Modèle Whisper chargé.")


app = Flask(__name__)
CORS(app)


def ffmpeg_to_wav16k_mono(input_path: Path, output_path: Path) -> None:
    """
    Convertit n'importe quel format audio (webm/mp3/m4a/wav...) en WAV 16kHz mono PCM.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(output_path),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.strip()}")


def transcribe_whisper(wav_path: Path, language: Optional[str] = "fr") -> dict:
    """
    Transcription avec faster-whisper.
    Retourne un dict avec texte, segments, langue et métadonnées.
    """
    segments, info = whisper_model.transcribe(
        str(wav_path),
        language=language,
        vad_filter=True,
        beam_size=5,
    )

    full_text_parts = []
    seg_list = []

    for s in segments:
        seg_text = (s.text or "").strip()
        if seg_text:
            full_text_parts.append(seg_text)

        seg_list.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": seg_text,
        })

    full_text = " ".join(full_text_parts).strip()

    return {
        "text": full_text,
        "segments": seg_list,
        "language": info.language,
        "language_probability": float(info.language_probability) if info.language_probability is not None else 0.0,
        "duration": float(info.duration) if info.duration is not None else 0.0,
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "whisper_model": WHISPER_SIZE,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE_TYPE,
    }), 200


@app.route("/stt", methods=["POST"])
def stt():
    if "audio" not in request.files:
        return jsonify({
            "status": "error",
            "error": "missing 'audio' file field"
        }), 400

    audio_file = request.files["audio"]
    lang = request.form.get("lang", "fr")

    uid = uuid.uuid4().hex
    filename = audio_file.filename or "audio.webm"

    in_path = TMP_DIR / f"{uid}_{filename}"
    wav_path = TMP_DIR / f"{uid}.wav"

    try:
        audio_file.save(str(in_path))

        ffmpeg_to_wav16k_mono(in_path, wav_path)

        result = transcribe_whisper(wav_path, language=lang)

        return jsonify({
            "status": "ok",
            **result,
        }), 200

    except Exception as e:
        print("[ERROR /stt]", str(e))
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    finally:
        for path in [in_path, wav_path]:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
            
            
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=APP_PORT, debug=True, use_reloader=False)