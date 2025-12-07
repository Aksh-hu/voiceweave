import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*80)
print("VOICEWEAVE: DEEP MECHANISTIC ANALYSIS FOR MIT CCC")
print("="*80)

# Load merged data
df = pd.read_csv("D:/VoiceWeave_MPDD/data/merged_utterances.csv")
print(f"\nTotal utterances: {len(df)}")

# Binary classification: suppressed vs amplified
df_model = df[df['voice_status'].isin(['suppressed', 'amplified'])].copy()
df_model = df_model.reset_index(drop=True)
print(f"Model dataset size: {len(df_model)}")

# ==================== FEATURE ENGINEERING (NO EMOTION) ====================
print("\n== Feature Engineering (Conversational Context Only) ==")

# Calculate conversational features
df_model['utterance_len'] = df_model['utterance'].str.len()
df_model['word_count'] = df_model['utterance'].str.split().str.len()

# Speaker diversity per dialogue
speaker_counts = df_model.groupby('dialogue_id')['speaker'].nunique().to_dict()
df_model['speakers_in_dialogue'] = df_model['dialogue_id'].map(speaker_counts)

# Turn position features
df_model['is_first_turn'] = (df_model['turn_index'] == 0).astype(int)
df_model['is_early_turn'] = (df_model['turn_index'] <= 2).astype(int)

# Speaker activity (how many times this speaker spoke in dialogue)
speaker_activity = df_model.groupby(['dialogue_id', 'speaker']).size().to_dict()
df_model['speaker_turn_count'] = df_model.apply(
    lambda x: speaker_activity.get((x['dialogue_id'], x['speaker']), 1), axis=1
)

# Source indicator
df_model['is_mpdd'] = (df_model['source'] == 'MPDD').astype(int)

# Select features (NO EMOTION - pure conversational structure)
feature_cols = [
    'turn_index',
    'utterance_len',
    'word_count',
    'speakers_in_dialogue',
    'is_first_turn',
    'is_early_turn',
    'speaker_turn_count',
    'is_mpdd'
]

X = df_model[feature_cols].copy()
y = df_model['voice_status'].copy()

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Encode target
y_enc = (y == 'suppressed').astype(int)  # 1 = suppressed, 0 = amplified

# Train/test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y_enc, df_model.index, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"Train: {len(X_train)}   Test: {len(X_test)}")

# ==================== TRAIN RANDOM FOREST ====================
print("\n== Training Random Forest (Deep Mechanistic Model) ==")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['amplified', 'suppressed']))
auc = roc_auc_score(y_test, y_proba)
print(f"\nAUC-ROC: {auc:.4f}")

# ==================== VISUALIZATION 1: CLASS BALANCE ====================
print("\n== Creating Visualizations ==")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# By source
source_counts = df['voice_status'].groupby(df['source']).value_counts().unstack(fill_value=0)
source_counts.plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
axes[0].set_title('Voice Status Distribution by Source', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xlabel('Source', fontsize=12)
axes[0].legend(title='Voice Status')

# Overall
df['voice_status'].value_counts().plot(kind='bar', ax=axes[1], color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
axes[1].set_title('Overall Voice Status Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_xlabel('Voice Status', fontsize=12)

plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/class_balance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Class balance visualization saved")

# ==================== VISUALIZATION 2: CONFUSION MATRIX ====================
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['amplified', 'suppressed'],
            yticklabels=['amplified', 'suppressed'],
            cbar_kws={'label': 'Count'}, ax=ax, annot_kws={"size": 16})
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix: Deep Mechanistic Model', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/confusion_matrix_deep.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Confusion matrix saved")

# ==================== VISUALIZATION 3: FEATURE IMPORTANCE ====================
feat_importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_importance)))
feat_importance.plot(kind='barh', color=colors, ax=ax, edgecolor='black', linewidth=1.2)
ax.set_xlabel('Gini Importance', fontsize=14, fontweight='bold')
ax.set_ylabel('Features', fontsize=14, fontweight='bold')
ax.set_title('Feature Importances: Deep Mechanistic Model', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(feat_importance.values):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/feature_importance_deep.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature importance saved")

# ==================== SHAP MECHANISTIC ANALYSIS ====================
print("\n== Computing SHAP Values ==")
explainer = shap.TreeExplainer(rf)
shap_sample_size = min(1000, len(X_test))
X_shap = X_test.iloc[:shap_sample_size]
shap_values = explainer.shap_values(X_shap)

if isinstance(shap_values, list):
    sv = shap_values[1]  # Suppressed class
else:
    sv = shap_values

# SHAP Beeswarm
fig, ax = plt.subplots(figsize=(14, 10))
shap.summary_plot(sv, X_shap, feature_names=feature_cols, show=False)
plt.title('SHAP Summary: Mechanistic Feature Impact on Suppression', 
          fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/shap_deep_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ SHAP beeswarm saved")

# SHAP Bar
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(sv, X_shap, feature_names=feature_cols, plot_type="bar", show=False)
plt.title('SHAP Feature Importance: Mean Absolute Impact', 
          fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/shap_deep_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ SHAP bar plot saved")

# ==================== ABLATION STUDY ====================
print("\n== Running Ablation Study ==")
baseline_acc = accuracy_score(y_test, y_pred)
ablation_results = {'baseline': baseline_acc}

for feat in feature_cols:
    # Remove feature
    X_train_ablate = X_train.drop(columns=[feat])
    X_test_ablate = X_test.drop(columns=[feat])
    
    # Train and test
    rf_ablate = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_ablate.fit(X_train_ablate, y_train)
    y_pred_ablate = rf_ablate.predict(X_test_ablate)
    
    ablation_results[feat] = accuracy_score(y_test, y_pred_ablate)

# Plot ablation
ablation_df = pd.DataFrame(list(ablation_results.items()), columns=['Feature', 'Accuracy'])
ablation_df['Accuracy_Drop'] = baseline_acc - ablation_df['Accuracy']
ablation_df = ablation_df[ablation_df['Feature'] != 'baseline'].sort_values('Accuracy_Drop', ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
colors = ['red' if x > 0 else 'green' for x in ablation_df['Accuracy_Drop']]
ablation_df.plot(x='Feature', y='Accuracy_Drop', kind='barh', ax=ax, color=colors, legend=False)
ax.set_xlabel('Accuracy Drop (Causal Importance)', fontsize=14, fontweight='bold')
ax.set_ylabel('Feature Removed', fontsize=14, fontweight='bold')
ax.set_title('Ablation Study: Feature Causal Necessity', fontsize=16, fontweight='bold', pad=20)
ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('D:/VoiceWeave_MPDD/results/ablation_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Ablation study saved")

# ==================== SAVE PREDICTIONS ====================
df_test = df_model.loc[idx_test].copy()
df_test['predicted'] = ['suppressed' if p else 'amplified' for p in y_pred]
df_test['prob_suppressed'] = y_proba
df_test.to_csv('D:/VoiceWeave_MPDD/results/deep_mechanistic_predictions.csv', index=False, encoding='utf-8')
print("✓ Test predictions saved")

print("\n" + "="*80)
print("✅ DEEP MECHANISTIC ANALYSIS COMPLETE!")
print("="*80)
print("\nMIT-Level Outputs Generated:")
print("  1. Class balance: results/class_balance.png")
print("  2. Confusion matrix: results/confusion_matrix_deep.png")
print("  3. Feature importance: results/feature_importance_deep.png")
print("  4. SHAP beeswarm: results/shap_deep_beeswarm.png")
print("  5. SHAP bar: results/shap_deep_bar.png")
print("  6. Ablation study: results/ablation_study.png")
print("  7. Predictions: results/deep_mechanistic_predictions.csv")
print(f"\nModel Performance: AUC-ROC = {auc:.4f}")
print("\n🚀 Ready for counterfactual generation and Flask demo!")
