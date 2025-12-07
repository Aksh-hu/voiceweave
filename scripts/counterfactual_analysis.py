import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COUNTERFACTUAL INTERVENTION GENERATION")
print("="*80)

# Load data and model (retrain briefly to save)
df = pd.read_csv("D:/VoiceWeave_MPDD/data/merged_utterances.csv")
df_model = df[df['voice_status'].isin(['suppressed', 'amplified'])].copy().reset_index(drop=True)

# Recreate features (same as deep_mechanistic_analysis.py)
df_model['utterance_len'] = df_model['utterance'].str.len()
df_model['word_count'] = df_model['utterance'].str.split().str.len()
speaker_counts = df_model.groupby('dialogue_id')['speaker'].nunique().to_dict()
df_model['speakers_in_dialogue'] = df_model['dialogue_id'].map(speaker_counts)
df_model['is_first_turn'] = (df_model['turn_index'] == 0).astype(int)
df_model['is_early_turn'] = (df_model['turn_index'] <= 2).astype(int)
speaker_activity = df_model.groupby(['dialogue_id', 'speaker']).size().to_dict()
df_model['speaker_turn_count'] = df_model.apply(
    lambda x: speaker_activity.get((x['dialogue_id'], x['speaker']), 1), axis=1
)
df_model['is_mpdd'] = (df_model['source'] == 'MPDD').astype(int)

feature_cols = [
    'turn_index', 'utterance_len', 'word_count', 'speakers_in_dialogue',
    'is_first_turn', 'is_early_turn', 'speaker_turn_count', 'is_mpdd'
]

X = df_model[feature_cols].copy()
y = (df_model['voice_status'] == 'suppressed').astype(int)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Train RF (quick)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, df_model.index, test_size=0.2, random_state=42, stratify=y
)
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Get suppressed test examples
df_test = df_model.loc[idx_test].copy()
X_test_original = X.loc[idx_test].copy()
suppressed_idx = df_test[df_test['voice_status'] == 'suppressed'].index[:50]  # Top 50

print(f"\nAnalyzing {len(suppressed_idx)} suppressed examples...")

# Counterfactual generation
counterfactuals = []

for idx in suppressed_idx:
    original = X_scaled.loc[idx].values.reshape(1, -1)
    original_prob = rf.predict_proba(original)[0, 1]
    
    if original_prob < 0.5:  # Already amplified, skip
        continue
    
    best_intervention = None
    best_new_prob = original_prob
    
    # Try perturbing each feature
    for feat_idx, feat_name in enumerate(feature_cols):
        # Try decreasing feature value
        perturbed = original.copy()
        perturbed[0, feat_idx] -= 0.5  # One std deviation
        new_prob = rf.predict_proba(perturbed)[0, 1]
        
        if new_prob < 0.5 and new_prob < best_new_prob:  # Flipped to amplified
            best_new_prob = new_prob
            best_intervention = {
                'feature': feat_name,
                'direction': 'decrease',
                'original_value': X_test_original.loc[idx, feat_name],
                'prob_change': original_prob - new_prob
            }
        
        # Try increasing
        perturbed = original.copy()
        perturbed[0, feat_idx] += 0.5
        new_prob = rf.predict_proba(perturbed)[0, 1]
        
        if new_prob < 0.5 and new_prob < best_new_prob:
            best_new_prob = new_prob
            best_intervention = {
                'feature': feat_name,
                'direction': 'increase',
                'original_value': X_test_original.loc[idx, feat_name],
                'prob_change': original_prob - new_prob
            }
    
    if best_intervention:
        counterfactuals.append({
            'dialogue_id': df_test.loc[idx, 'dialogue_id'],
            'turn_index': df_test.loc[idx, 'turn_index'],
            'speaker': df_test.loc[idx, 'speaker'],
            'utterance': df_test.loc[idx, 'utterance'][:100],
            'original_prob': original_prob,
            'intervention_feature': best_intervention['feature'],
            'intervention_direction': best_intervention['direction'],
            'original_feature_value': best_intervention['original_value'],
            'prob_after_intervention': best_new_prob,
            'prob_reduction': best_intervention['prob_change']
        })

cf_df = pd.DataFrame(counterfactuals)
cf_df.to_csv('D:/VoiceWeave_MPDD/results/counterfactual_interventions.csv', index=False, encoding='utf-8')
print(f"\n✓ Generated {len(cf_df)} counterfactual interventions")

# Summary statistics
print("\n=== Counterfactual Summary ===")
print(f"Most impactful feature: {cf_df['intervention_feature'].mode()[0]}")
print(f"Average suppression reduction: {cf_df['prob_reduction'].mean():.3f}")
print("\nTop 5 Interventions:")
print(cf_df.nlargest(5, 'prob_reduction')[['speaker', 'intervention_feature', 'intervention_direction', 'prob_reduction']])

# Generate natural language explanations
print("\n=== Sample Intervention Recipes ===")
for i, row in cf_df.head(5).iterrows():
    feature = row['intervention_feature']
    direction = row['intervention_direction']
    
    # Map to actionable advice
    if feature == 'turn_index' and direction == 'decrease':
        advice = "🎯 Invite this speaker earlier in the conversation"
    elif feature == 'speaker_turn_count' and direction == 'increase':
        advice = "🎯 Give this speaker more opportunities to contribute"
    elif feature == 'is_first_turn' and direction == 'increase':
        advice = "🎯 Start the dialogue with this speaker's perspective"
    elif feature == 'utterance_len' and direction == 'increase':
        advice = "🎯 Encourage this speaker to elaborate more"
    elif feature == 'word_count' and direction == 'increase':
        advice = "🎯 Ask follow-up questions to draw out more detail"
    else:
        advice = f"🎯 Adjust {feature} ({direction})"
    
    print(f"\nDialogue {row['dialogue_id']}, Turn {row['turn_index']}")
    print(f"  Speaker: {row['speaker']}")
    print(f"  Current suppression risk: {row['original_prob']:.2%}")
    print(f"  {advice}")
    print(f"  → Expected risk after: {row['prob_after_intervention']:.2%}")

print("\n" + "="*80)
print("✅ COUNTERFACTUAL ANALYSIS COMPLETE!")
print("="*80)
print("\nOutputs:")
print("  - Counterfactual recipes: results/counterfactual_interventions.csv")
print("\n🚀 Ready for LLM-powered intervention generation!")
