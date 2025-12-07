import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADVANCED COUNTERFACTUAL V2: Enhanced Coverage (Target: 50%+)")
print("="*80)

try:
    # Load and prepare data
    df = pd.read_csv("D:/VoiceWeave_MPDD/data/merged_utterances.csv")
    df_model = df[df['voice_status'].isin(['suppressed', 'amplified'])].copy().reset_index(drop=True)
    
    print(f"Loaded {len(df_model)} utterances")
    
    # Feature engineering
    print("\n=== Feature Engineering ===")
    df_model['utterance_len'] = df_model['utterance'].fillna('').str.len()
    df_model['word_count'] = df_model['utterance'].fillna('').str.split().str.len()
    
    speaker_counts = df_model.groupby('dialogue_id')['speaker'].nunique().to_dict()
    df_model['speakers_in_dialogue'] = df_model['dialogue_id'].map(speaker_counts).fillna(1)
    
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
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)
    
    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, df_model.index, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("\n=== Training Random Forest ===")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"Model trained. Test accuracy: {rf.score(X_test, y_test):.3f}")
    
    # Get suppressed test cases
    df_test = df_model.loc[idx_test].copy()
    suppressed_mask = df_test['voice_status'] == 'suppressed'
    num_suppressed = suppressed_mask.sum()
    
    sample_size = min(200, num_suppressed)
    suppressed_idx = df_test[suppressed_mask].sample(sample_size, random_state=42).index
    
    print(f"\nAnalyzing {len(suppressed_idx)} suppressed examples...")
    print("Using enhanced search with wider perturbations and relaxed threshold")
    
    # Enhanced counterfactual search
    multi_cf_results = []
    
    # Expanded perturbation magnitudes
    magnitudes = [-2.0, -1.5, -1.0, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    # Relaxed success threshold
    SUCCESS_THRESHOLD = 0.45  # Was 0.5
    
    for i, idx in enumerate(suppressed_idx):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(suppressed_idx)} ({len(multi_cf_results)} found, {len(multi_cf_results)/(i+1)*100:.1f}% coverage so far)...")
        
        original = X_scaled.loc[idx].values.reshape(1, -1)
        original_prob = rf.predict_proba(original)[0, 1]
        
        if original_prob < 0.5:
            continue
        
        best_intervention = None
        best_prob = original_prob
        found = False
        
        # PHASE 1: Systematic 2-feature grid search
        for feat1, feat2 in combinations(range(len(feature_cols)), 2):
            if found:
                break
            # Test all sign/magnitude combinations
            for delta1, delta2 in product(magnitudes[:8], magnitudes[4:]):  # Optimize range
                perturbed = original.copy()
                perturbed[0, feat1] += delta1
                perturbed[0, feat2] += delta2
                new_prob = rf.predict_proba(perturbed)[0, 1]
                
                if new_prob < SUCCESS_THRESHOLD and new_prob < best_prob:
                    best_prob = new_prob
                    best_intervention = {
                        'features': (feature_cols[feat1], feature_cols[feat2]),
                        'changes': (delta1, delta2),
                        'prob_reduction': original_prob - new_prob
                    }
                    if new_prob < 0.4:  # Strong flip, stop early
                        found = True
                        break
        
        # PHASE 2: Full 3-feature search (if not found yet)
        if not found:
            for feat_combo in combinations(range(len(feature_cols)), 3):
                if found:
                    break
                # Sample magnitude combinations
                for deltas in [
                    (-1.5, 1.5, -0.5), (-2.0, 2.0, -1.0), (-1.0, 1.0, 0.5),
                    (-0.7, 0.7, -0.5), (-1.5, 0.5, 1.5), (0.7, -0.7, 1.0),
                    (-2.0, 1.0, 1.0), (1.5, -1.5, 0.5), (-1.0, 2.0, -2.0)
                ]:
                    perturbed = original.copy()
                    perturbed[0, feat_combo[0]] += deltas[0]
                    perturbed[0, feat_combo[1]] += deltas[1]
                    perturbed[0, feat_combo[2]] += deltas[2]
                    new_prob = rf.predict_proba(perturbed)[0, 1]
                    
                    if new_prob < SUCCESS_THRESHOLD and new_prob < best_prob:
                        best_prob = new_prob
                        best_intervention = {
                            'features': tuple(feature_cols[f] for f in feat_combo),
                            'changes': deltas,
                            'prob_reduction': original_prob - new_prob
                        }
                        if new_prob < 0.4:
                            found = True
                            break
        
        # PHASE 3: 4-feature combinations for hard cases
        if not found and original_prob > 0.7:  # Very confident suppression
            four_combos = list(combinations(range(len(feature_cols)), 4))[:30]  # Sample
            for feat_combo in four_combos:
                perturbed = original.copy()
                # Apply strong perturbations
                perturbed[0, feat_combo[0]] -= 1.5
                perturbed[0, feat_combo[1]] += 1.5
                perturbed[0, feat_combo[2]] -= 1.0
                perturbed[0, feat_combo[3]] += 1.0
                new_prob = rf.predict_proba(perturbed)[0, 1]
                
                if new_prob < SUCCESS_THRESHOLD and new_prob < best_prob:
                    best_prob = new_prob
                    best_intervention = {
                        'features': tuple(feature_cols[f] for f in feat_combo),
                        'changes': (-1.5, 1.5, -1.0, 1.0),
                        'prob_reduction': original_prob - new_prob
                    }
                    break
        
        if best_intervention:
            multi_cf_results.append({
                'dialogue_id': df_test.loc[idx, 'dialogue_id'],
                'turn_index': df_test.loc[idx, 'turn_index'],
                'speaker': df_test.loc[idx, 'speaker'],
                'utterance': str(df_test.loc[idx, 'utterance'])[:80],
                'original_prob': original_prob,
                'intervention_features': ' + '.join(best_intervention['features']),
                'new_prob': best_prob,
                'prob_reduction': best_intervention['prob_reduction'],
                'num_features': len(best_intervention['features'])
            })
    
    if len(multi_cf_results) == 0:
        print("\n⚠️  No counterfactuals found even with enhanced search.")
        exit(0)
    
    cf_df = pd.DataFrame(multi_cf_results)
    cf_df.to_csv('D:/VoiceWeave_MPDD/results/multi_feature_counterfactuals_v2.csv', index=False, encoding='utf-8')
    
    coverage = len(cf_df) / len(suppressed_idx) * 100
    
    print(f"\n✓ Found {len(cf_df)} counterfactual interventions")
    print(f"  Coverage: {coverage:.1f}% (Target: 50%+)")
    print(f"  Average suppression reduction: {cf_df['prob_reduction'].mean():.3f}")
    print(f"\nIntervention complexity:")
    print(cf_df['num_features'].value_counts().sort_index())
    
    print("\n=== Top 5 Most Effective Interventions ===")
    top5 = cf_df.nlargest(5, 'prob_reduction')
    for i, row in top5.iterrows():
        print(f"\nDialogue {row['dialogue_id']}, Turn {row['turn_index']}")
        print(f"  Speaker: {row['speaker']}")
        print(f"  Intervention: {row['intervention_features']}")
        print(f"  Impact: {row['original_prob']:.2%} → {row['new_prob']:.2%} (-{row['prob_reduction']:.2%})")
    
    print("\n" + "="*80)
    print("✅ ENHANCED COUNTERFACTUAL ANALYSIS COMPLETE!")
    print("="*80)
    
    if coverage >= 50:
        print(f"🎯 TARGET ACHIEVED: {coverage:.1f}% coverage!")
    else:
        print(f"Coverage: {coverage:.1f}% (Improved from 19.5%)")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
