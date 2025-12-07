import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load data
df = pd.read_csv("data/merged_utterances.csv")

# Keep only amplified/suppressed
df = df[df["voice_status"].isin(["suppressed", "amplified"])].copy()
df = df.reset_index(drop=True)

# Structural features
df["utterance_len"] = df["utterance"].astype(str).str.len()
df["word_count"] = df["utterance"].astype(str).str.split().str.len()
df["speakers_in_dialogue"] = df.groupby("dialogue_id")["speaker"].transform("nunique")
df["speaker_turn_count"] = df.groupby(["dialogue_id", "speaker"])["utterance"].transform("count")

X = df[["utterance_len", "word_count", "speakers_in_dialogue", "speaker_turn_count"]].copy()
y = (df["voice_status"] == "suppressed").astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")

# Scale all 4 features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RF
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
y_proba = rf.predict_proba(X_test_scaled)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["amplified", "suppressed"]))
print("\nAUC-ROC:", roc_auc_score(y_test, y_proba))

# Save model and scaler
os.makedirs("models", exist_ok=True)
with open("models/rf_suppression_model.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved rf_suppression_model.pkl and scaler.pkl with 4 structural features.")
