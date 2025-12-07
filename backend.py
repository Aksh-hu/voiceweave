from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import logging

from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voiceweave-backend")

app = Flask(__name__)
CORS(app)

print("\n" + "=" * 78)
print("VOICEWEAVE BACKEND – INITIALIZING")
print("=" * 78)

rf_model = None
scaler = None

try:
    with open(os.path.join("models", "rf_suppression_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    print(f"[✓] Random Forest loaded, expects {rf_model.n_features_in_} features")
except Exception as e:
    print(f"[✗] RF load error: {e}")

try:
    with open(os.path.join("models", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    print(f"[✓] Scaler loaded, expects {scaler.n_features_in_} features")
except Exception as e:
    print(f"[✗] Scaler load error: {e}")

print("=" * 78)
print("BACKEND READY – http://0.0.0.0:5000")
print("=" * 78 + "\n")


def build_feature_frame(lines):
    """Parse speaker|utterance lines into feature DataFrame."""
    dialogue_rows = []
    for line in lines:
        if "|" not in line:
            continue
        speaker, text = line.split("|", 1)
        speaker = speaker.strip()
        text = text.strip()
        if not speaker or not text:
            continue
        dialogue_rows.append({"speaker": speaker, "utterance": text})

    if len(dialogue_rows) < 2:
        return None, "Need at least 2 turns in format 'speaker|text'."

    df = pd.DataFrame(dialogue_rows)
    df["dialogue_id"] = 0  # single dialogue context

    df["utterance_len"] = df["utterance"].astype(str).str.len()
    df["word_count"] = df["utterance"].astype(str).str.split().str.len()
    df["speakers_in_dialogue"] = df["speaker"].nunique()
    df["speaker_turn_count"] = df.groupby("speaker")["utterance"].transform("count")

    feature_cols = [
        "utterance_len",
        "word_count",
        "speakers_in_dialogue",
        "speaker_turn_count",
    ]
    X = df[feature_cols].fillna(0).astype(float).values
    return df, X


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "online",
            "rf_ready": rf_model is not None,
            "scaler_ready": scaler is not None,
        }
    ), 200


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if rf_model is None or scaler is None:
            return jsonify({"error": "Models not loaded on backend."}), 500

        data = request.get_json(force=True) or {}
        transcript_text = (data.get("transcript") or "").strip()
        if not transcript_text:
            return jsonify({"error": "Transcript is empty."}), 400

        lines = [l.strip() for l in transcript_text.split("\n") if l.strip()]
        df, X = build_feature_frame(lines)
        if df is None:
            return jsonify({"error": X}), 400

        if X.shape[1] != rf_model.n_features_in_ or X.shape[1] != scaler.n_features_in_:
            msg = (
                f"Feature mismatch: X has {X.shape[1]}, "
                f"RF expects {rf_model.n_features_in_}, "
                f"scaler expects {scaler.n_features_in_}."
            )
            return jsonify({"error": msg}), 500

        X_scaled = scaler.transform(X)
        probs = rf_model.predict_proba(X_scaled)[:, 1]

        turns = []
        recommendations = []
        for i, row in df.reset_index(drop=True).iterrows():
            p = float(probs[i])
            status = "suppressed" if p > 0.5 else "amplified"

            turns.append(
                {
                    "turn_index": int(i),
                    "speaker": row["speaker"],
                    "text": row["utterance"][:200],
                    "suppression_prob": p,
                    "status": status,
                }
            )

            if p > 0.65:
                recommendations.append(
                    {
                        "turn_index": int(i),
                        "speaker": row["speaker"],
                        "risk": "high" if p > 0.80 else "moderate",
                        "probability": p,
                        "suggestion": (
                            f"Invite {row['speaker']} to expand on this turn. "
                            "Their contribution is at elevated risk of being suppressed."
                        ),
                    }
                )

        response = {
            "success": True,
            "summary": {
                "total_turns": int(len(df)),
                "unique_speakers": int(df["speaker"].nunique()),
                "avg_risk": float(np.mean(probs)),
                "high_risk_count": int(sum(p > 0.65 for p in probs)),
            },
            "turns": turns,
            "recommendations": recommendations,
        }
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Error in /api/analyze")
        return jsonify({"error": f"Analysis failed: {e}"}), 500


@app.route("/api/example", methods=["GET"])
def example():
    large_example = (
        "Facilitator|Thanks everyone for joining. The goal today is to discuss our new community program.\n"
        "Alex|I think the budget is tight, and we should start small.\n"
        "Priya|I am worried that if we start small, some neighborhoods will be left out.\n"
        "Ravi|Maybe we can rotate the pilot locations every few months.\n"
        "Alex|Rotations are fine, but we still do not have clarity on who decides the order.\n"
        "Facilitator|That is a good point. Who feels their perspective has not been heard yet?\n"
        "Sara|I have been quiet because I am new here, but I think residents should nominate pilot sites.\n"
        "Alex|If residents nominate, we might get pressure from louder groups.\n"
        "Priya|That is exactly why we need a clear process so quieter groups are not sidelined.\n"
        "Facilitator|Thank you. Let us slow down and hear from people who have not spoken much yet.\n"
    )
    return jsonify({"example": large_example}), 200


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "VoiceWeave backend running"}), 200


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
