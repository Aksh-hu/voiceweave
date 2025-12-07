import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load merged data
df = pd.read_csv("D:/VoiceWeave_MPDD/data/merged_utterances.csv")

# Only binary target (MIT-level clarity)
df_model = df[df['voice_status'].isin(['suppressed', 'amplified'])].copy()
df_model = df_model.reset_index(drop=True)

# Encode features
X = df_model[['emotion', 'speaker', 'turn_index', 'source']].copy()
y = df_model['voice_status'].copy()

cat_cols = ['emotion', 'speaker', 'source']
for col in cat_cols:
    enc = LabelEncoder()
    X[col] = enc.fit_transform(X[col].astype(str))

scaler = StandardScaler()
X['turn_index'] = scaler.fit_transform(X[['turn_index']])

y_enc = LabelEncoder().fit_transform(y)  # 0 = amplified, 1 = suppressed

# Train/test split (save indices for robust downstream referencing)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y_enc, df_model.index, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")

# Train RF
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['amplified', 'suppressed']))
print("\nAUC-ROC:", roc_auc_score(y_test, y_proba))

# SHAP mechanistic analysis
print("\n== SHAP summary ==")
explainer = shap.TreeExplainer(rf)
shap_vals = explainer.shap_values(X_test)

if isinstance(shap_vals, list) and len(shap_vals) == 2:
    sv = shap_vals[1]  # For suppressed
elif isinstance(shap_vals, list):
    sv = shap_vals[0]
else:
    sv = shap_vals

print(f"SHAP values shape: {sv.shape}, X_test shape: {X_test.shape} (should match: [samples, features])")

# SHAP Beeswarm
shap.summary_plot(sv, X_test, feature_names=X.columns.tolist(), show=False)
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/shap_merged_beeswarm.png', dpi=300)
plt.close()

# Feature importances
feat_importance = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 5))
feat_importance.sort_values().plot(kind='barh', color='orange')
plt.title('Feature Importance (RF, merged corpus)', fontsize=16)
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/feature_importance_merged.png', dpi=300)
plt.close()

# Save predictions
df_test = df_model.loc[idx_test].copy()
df_test['predicted'] = ['suppressed' if p else 'amplified' for p in y_pred]
df_test['prob_suppressed'] = y_proba
df_test.to_csv('D:/VoiceWeave_MPDD/results/merged_rf_test_predictions.csv', index=False)

print("Done: Beeswarm, predictions, and feature importances saved.")

print("\nSample predictions:")
print(df_test[['utterance','emotion','predicted','prob_suppressed']].head())
