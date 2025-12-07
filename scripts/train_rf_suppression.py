import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load labeled data
df = pd.read_csv('D:/VoiceWeave_MPDD/data/mpdd_labeled.csv')

# Use only 'suppressed' and 'amplified' as targets for binary classification
df = df[df['voice_status'].isin(['suppressed', 'amplified'])].copy()
df = df.reset_index(drop=True)

print(f"Total samples: {len(df)}")
print(f"Suppressed: {(df['voice_status'] == 'suppressed').sum()}")
print(f"Amplified: {(df['voice_status'] == 'amplified').sum()}")

# Feature Engineering
feat_columns = [
    'speaker_position',
    'emotion',
    'num_listeners',
    'utterance_len',
    'utterance_word_count',
    'prev_emotion',
    'next_emotion',
    'suppression_score'
]

X = df[feat_columns].copy()
y = df['voice_status'].copy()

# Encode categorical features
cat_columns = ['speaker_position', 'emotion', 'prev_emotion', 'next_emotion']
encoders = {}
for col in cat_columns:
    enc = LabelEncoder()
    X[col] = enc.fit_transform(X[col].fillna('none'))
    encoders[col] = enc

# Scale numeric features
num_columns = ['num_listeners', 'utterance_len', 'utterance_word_count', 'suppression_score']
scaler = StandardScaler()
X[num_columns] = scaler.fit_transform(X[num_columns])

# Encode target
label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y_enc, df.index, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train Random Forest
print("\n=== Training Random Forest Classifier ===")
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

# AUC-ROC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC Score: {auc_score:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n=== Confusion Matrix ===")
print(cm)

# Plot confusion matrix (improved)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=label_enc.classes_, yticklabels=label_enc.classes_,
            cbar_kws={'label': 'Count'}, ax=ax, annot_kws={"size": 16})
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix: Voice Suppression Detection', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix saved")
plt.close()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, color='darkblue', lw=3, label=f'ROC curve (AUC = {auc_score:.3f})')
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('ROC Curve: Suppression Detection Performance', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ ROC curve saved")
plt.close()

# Feature importances (improved visualization)
feat_importance = pd.Series(rf.feature_importances_, index=feat_columns).sort_values(ascending=True)
print("\n=== Random Forest Feature Importances ===")
print(feat_importance.sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_importance)))
feat_importance.plot(kind='barh', color=colors, ax=ax, edgecolor='black', linewidth=1.2)
ax.set_xlabel('Gini Importance', fontsize=14, fontweight='bold')
ax.set_ylabel('Features', fontsize=14, fontweight='bold')
ax.set_title('Feature Importances: Voice Suppression Classifier', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(feat_importance.values):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance plot saved")
plt.close()

# SHAP Interpretability (PUBLICATION QUALITY)
print("\n=== Computing SHAP Values (Mechanistic Interpretability) ===")
shap_sample_size = min(1000, len(X_test))
X_shap = X_test.iloc[:shap_sample_size].copy()
X_shap.columns = feat_columns

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_shap)

# Handle different SHAP output formats
if isinstance(shap_values, list) and len(shap_values) == 2:
    # Binary classification: [class_0_shap, class_1_shap]
    shap_vals_suppressed = shap_values[1]
    base_value = explainer.expected_value[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    # Shape (n_samples, n_features, n_classes)
    shap_vals_suppressed = shap_values[:, :, 1]
    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
else:
    shap_vals_suppressed = shap_values
    base_value = explainer.expected_value

print(f"SHAP values shape: {shap_vals_suppressed.shape}")
print(f"Using {shap_sample_size} samples for SHAP analysis")

# SHAP Summary Plot (Beeswarm) - ENHANCED
fig, ax = plt.subplots(figsize=(14, 10))
shap.summary_plot(shap_vals_suppressed, X_shap, feature_names=feat_columns, 
                  show=False, plot_size=(14, 10))
plt.title('SHAP Summary: Feature Impact on Suppression Prediction', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
print("✓ SHAP beeswarm plot saved")
plt.close()

# SHAP Bar Plot (Mean Absolute) - ENHANCED
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_vals_suppressed, X_shap, feature_names=feat_columns, 
                  plot_type="bar", show=False)
plt.title('SHAP Feature Importance: Mean Absolute Impact', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Mean |SHAP Value|', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/shap_summary_bar.png', dpi=300, bbox_inches='tight')
print("✓ SHAP bar plot saved")
plt.close()

# SHAP Waterfall Plot for Individual Example (FIXED)
sample_idx = 0
shap_explanation = shap.Explanation(
    values=shap_vals_suppressed[sample_idx], 
    base_values=base_value,
    data=X_shap.iloc[sample_idx].values,
    feature_names=feat_columns
)
fig, ax = plt.subplots(figsize=(12, 8))
shap.waterfall_plot(shap_explanation, show=False)
plt.title('SHAP Waterfall: Individual Turn Mechanistic Breakdown', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/shap_waterfall_example.png', dpi=300, bbox_inches='tight')
print("✓ SHAP waterfall plot saved")
plt.close()

# SHAP Dependence Plot (shows interaction between top 2 features)
top_features = feat_importance.sort_values(ascending=False).head(2).index.tolist()
fig, ax = plt.subplots(figsize=(12, 8))
shap.dependence_plot(
    top_features[0], 
    shap_vals_suppressed, 
    X_shap, 
    feature_names=feat_columns,
    interaction_index=top_features[1],
    show=False
)
plt.title(f'SHAP Dependence: {top_features[0]} (colored by {top_features[1]})', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/shap_dependence_plot.png', dpi=300, bbox_inches='tight')
print("✓ SHAP dependence plot saved")
plt.close()

# Save predictions
df_test = df.loc[idx_test].copy()
df_test['predicted'] = label_enc.inverse_transform(y_pred)
df_test['pred_proba_suppressed'] = y_pred_proba
df_test['correct'] = (y_test == y_pred)
df_test.to_csv('D:/VoiceWeave_MPDD/results/rf_test_predictions.csv', index=False, encoding='utf-8')
print("\n✓ Test predictions saved")

# Sample outputs
print("\n=== Sample Correct Suppressed Predictions ===")
correct_suppressed = df_test[(df_test['correct']) & (df_test['voice_status'] == 'suppressed')].head(3)
for idx, row in correct_suppressed.iterrows():
    print(f"\nDialogue {row['dialogue_id']}, Turn {row['turn_index']}")
    print(f"  Speaker: {row['speaker']} (Position: {row['speaker_position']})")
    print(f"  Emotion: {row['emotion']} → Next: {row['next_emotion']}")
    print(f"  Suppression Probability: {row['pred_proba_suppressed']:.3f}")
    print(f"  Predicted: {row['predicted']} ✓")

print("\n" + "="*80)
print("✅ TRAINING & MECHANISTIC ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated MIT-Level Outputs:")
print("  1. Confusion Matrix (300 DPI): results/confusion_matrix.png")
print("  2. ROC Curve with AUC: results/roc_curve.png")
print("  3. Feature Importance: results/feature_importance.png")
print("  4. SHAP Beeswarm (mechanistic): results/shap_summary_beeswarm.png")
print("  5. SHAP Bar Plot: results/shap_summary_bar.png")
print("  6. SHAP Waterfall (individual): results/shap_waterfall_example.png")
print("  7. SHAP Dependence Plot: results/shap_dependence_plot.png")
print("  8. Test Predictions CSV: results/rf_test_predictions.csv")
print(f"\nModel Performance: AUC-ROC = {auc_score:.4f}")
print("\n🚀 Ready for Flask demo and intervention generation!")
