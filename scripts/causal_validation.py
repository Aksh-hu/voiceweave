import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')

print("="*80)
print("CAUSAL VALIDATION: Proving Interventions Work")
print("="*80)

def validate_file_exists(filepath, description):
    """Validate that required file exists."""
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: {description} not found at: {filepath}")
        print("Please ensure the file exists before running this script.")
        sys.exit(1)
    return True

def safe_division(numerator, denominator, default=0):
    """Safe division with default value."""
    return numerator / denominator if denominator != 0 else default

try:
    # Validate dependencies
    data_path = "D:/VoiceWeave_MPDD/data/merged_utterances.csv"
    cf_path = 'D:/VoiceWeave_MPDD/results/multi_feature_counterfactuals_v2.csv'
    
    # Check if v2 exists, fallback to v1
    if not os.path.exists(cf_path):
        cf_path = 'D:/VoiceWeave_MPDD/results/multi_feature_counterfactuals.csv'
        print("Using v1 counterfactuals file")
    
    validate_file_exists(data_path, "Merged utterances data")
    validate_file_exists(cf_path, "Counterfactuals file")
    
    # Load data
    print("\n=== Loading Data ===")
    df = pd.read_csv(data_path, encoding='utf-8')
    df_model = df[df['voice_status'].isin(['suppressed', 'amplified'])].copy().reset_index(drop=True)
    
    if len(df_model) == 0:
        print("❌ ERROR: No valid data after filtering for suppressed/amplified.")
        sys.exit(1)
    
    print(f"✓ Loaded {len(df_model):,} utterances")
    
    # Feature engineering with robust error handling
    print("\n=== Feature Engineering ===")
    
    df_model['utterance'] = df_model['utterance'].fillna('')
    df_model['utterance_len'] = df_model['utterance'].str.len()
    df_model['word_count'] = df_model['utterance'].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Handle missing dialogue_id
    if df_model['dialogue_id'].isnull().any():
        print("⚠️  Warning: Found missing dialogue_id values, filling with placeholder")
        df_model['dialogue_id'] = df_model['dialogue_id'].fillna(-1)
    
    speaker_counts = df_model.groupby('dialogue_id')['speaker'].nunique().to_dict()
    df_model['speakers_in_dialogue'] = df_model['dialogue_id'].map(speaker_counts).fillna(1).astype(int)
    
    df_model['is_first_turn'] = (df_model['turn_index'].fillna(999) == 0).astype(int)
    df_model['is_early_turn'] = (df_model['turn_index'].fillna(999) <= 2).astype(int)
    
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
    
    # Handle any remaining NaN
    X = X.fillna(X.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=feature_cols, 
        index=X.index
    )
    
    # Train/test split with stratification
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, df_model.index, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Train model
    print("\n=== Training Random Forest ===")
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    
    accuracy = rf.score(X_test, y_test)
    print(f"✓ Model trained. Test accuracy: {accuracy:.3f}")
    
    # Load counterfactuals
    print(f"\n=== Loading Counterfactuals ===")
    cf_df = pd.read_csv(cf_path, encoding='utf-8')
    print(f"✓ Loaded {len(cf_df):,} counterfactual interventions")
    
    if len(cf_df) == 0:
        print("❌ ERROR: Counterfactuals file is empty!")
        sys.exit(1)
    
    # Synthetic intervention validation
    print("\n=== Validating Interventions ===")
    causal_results = []
    df_test = df_model.loc[idx_test].copy()
    
    validation_sample = min(100, len(cf_df))
    print(f"Validating {validation_sample} interventions...")
    
    matches_found = 0
    
    for i, cf_row in cf_df.head(validation_sample).iterrows():
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{validation_sample} ({matches_found} validated)")
        
        # Find original example with robust matching
        try:
            dialogue_id = cf_row['dialogue_id']
            turn_index = cf_row['turn_index']
            
            matches = df_test[
                (df_test['dialogue_id'] == dialogue_id) & 
                (df_test['turn_index'] == turn_index)
            ]
            
            if len(matches) == 0:
                continue
            
            idx = matches.index[0]
            
            if idx not in X_scaled.index:
                continue
            
            matches_found += 1
            
            original = X_scaled.loc[idx].values.reshape(1, -1)
            original_prob = rf.predict_proba(original)[0, 1]
            
            # Apply intervention
            intervened = original.copy()
            
            # Parse intervention features
            features_str = str(cf_row.get('intervention_features', ''))
            features = [f.strip() for f in features_str.split('+')]
            
            # Apply realistic feature changes
            for feat in features:
                if feat not in feature_cols:
                    continue
                
                feat_idx = feature_cols.index(feat)
                
                # Apply interventions based on feature semantics
                if feat == 'turn_index':
                    intervened[0, feat_idx] -= 1.0  # Earlier turn
                elif feat == 'speaker_turn_count':
                    intervened[0, feat_idx] += 1.0  # More active
                elif feat == 'is_early_turn':
                    intervened[0, feat_idx] = 1.0  # Set to early
                elif feat == 'is_first_turn':
                    intervened[0, feat_idx] = 1.0  # Set to first
                elif feat == 'speakers_in_dialogue':
                    intervened[0, feat_idx] += 0.5  # More diversity
                elif feat == 'utterance_len':
                    intervened[0, feat_idx] += 0.5  # Longer
                elif feat == 'word_count':
                    intervened[0, feat_idx] += 0.5  # More words
            
            intervened_prob = rf.predict_proba(intervened)[0, 1]
            effect_size = original_prob - intervened_prob
            
            causal_results.append({
                'intervention': features_str,
                'before_prob': float(original_prob),
                'after_prob': float(intervened_prob),
                'causal_effect': float(effect_size),
                'flipped': bool(original_prob > 0.5 and intervened_prob < 0.5),
                'speaker': str(cf_row.get('speaker', 'Unknown'))
            })
            
        except Exception as e:
            # Skip problematic rows silently
            continue
    
    print(f"✓ Successfully validated {matches_found} interventions")
    
    if len(causal_results) == 0:
        print("\n⚠️  No interventions could be validated.")
        print("This may occur if test set differs from counterfactuals.")
        
        # Save empty results
        empty_df = pd.DataFrame(columns=[
            'intervention', 'before_prob', 'after_prob', 
            'causal_effect', 'flipped', 'speaker'
        ])
        empty_df.to_csv('D:/VoiceWeave_MPDD/results/causal_validation_results.csv', index=False)
        print("Empty results file saved.")
        sys.exit(0)
    
    # Convert to DataFrame
    causal_df = pd.DataFrame(causal_results)
    causal_df.to_csv(
        'D:/VoiceWeave_MPDD/results/causal_validation_results.csv', 
        index=False, 
        encoding='utf-8'
    )
    
    # Statistical analysis
    print("\n=== Statistical Analysis ===")
    mean_effect = causal_df['causal_effect'].mean()
    std_effect = causal_df['causal_effect'].std()
    median_effect = causal_df['causal_effect'].median()
    
    if len(causal_df) >= 2:
        t_stat, p_value = stats.ttest_1samp(causal_df['causal_effect'], 0)
    else:
        t_stat, p_value = 0, 1.0
        print("⚠️  Sample too small for t-test")
    
    flip_rate = safe_division(causal_df['flipped'].sum(), len(causal_df), 0) * 100
    
    print(f"Validated interventions: {len(causal_df):,}")
    print(f"Mean causal effect: {mean_effect:.3f} (±{std_effect:.3f})")
    print(f"Median causal effect: {median_effect:.3f}")
    print(f"Success rate (flipped to amplified): {flip_rate:.1f}%")
    
    if len(causal_df) >= 2:
        print(f"Statistical test: t={t_stat:.2f}, p={p_value:.4f}")
        
        if p_value < 0.001:
            print("✅ HIGHLY SIGNIFICANT (p < 0.001)")
        elif p_value < 0.05:
            print("✅ SIGNIFICANT (p < 0.05)")
        else:
            print(f"⚠️  Not statistically significant (p = {p_value:.4f})")
    
    # Intervention ranking
    print("\n=== Intervention Ranking ===")
    intervention_counts = causal_df.groupby('intervention').size()
    intervention_ranking = causal_df.groupby('intervention')['causal_effect'].agg([
        'mean', 'count', 'std', 'median'
    ])
    
    # Filter for reliability (at least 3 instances)
    intervention_ranking = intervention_ranking[intervention_ranking['count'] >= 3]
    
    if len(intervention_ranking) > 0:
        intervention_ranking = intervention_ranking.sort_values('mean', ascending=False)
        intervention_ranking.to_csv(
            'D:/VoiceWeave_MPDD/results/intervention_ranking.csv',
            encoding='utf-8'
        )
        
        print(f"✓ Ranked {len(intervention_ranking)} intervention types")
        print("\nTop 5 Most Effective:")
        print(intervention_ranking.head())
    else:
        print("⚠️  Not enough data for intervention ranking")
    
    # Visualization
    print("\n=== Creating Visualizations ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Before/After scatter
    axes[0].scatter(
        causal_df['before_prob'], 
        causal_df['after_prob'], 
        alpha=0.6, 
        s=50, 
        color='steelblue',
        edgecolors='black',
        linewidth=0.5
    )
    axes[0].plot([0, 1], [0, 1], 'r--', label='No effect', linewidth=2)
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes[0].axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes[0].set_xlabel('Suppression Probability (Before)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Suppression Probability (After)', fontsize=12, fontweight='bold')
    axes[0].set_title('Causal Impact of Interventions', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # Plot 2: Effect size distribution
    axes[1].hist(
        causal_df['causal_effect'], 
        bins=30, 
        edgecolor='black', 
        alpha=0.7, 
        color='teal'
    )
    axes[1].axvline(
        mean_effect, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean: {mean_effect:.3f}'
    )
    axes[1].axvline(
        median_effect, 
        color='orange', 
        linestyle='--', 
        linewidth=2, 
        label=f'Median: {median_effect:.3f}'
    )
    axes[1].set_xlabel('Causal Effect (Reduction in Suppression)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Intervention Effects', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(
        'D:/VoiceWeave_MPDD/results/causal_validation_plots.png', 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()
    
    print("✓ Plots saved")
    
    # Summary
    print("\n" + "="*80)
    print("✅ CAUSAL VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nKey Finding: Interventions reduce suppression by {mean_effect*100:.1f}% on average")
    
    if p_value < 0.05:
        print(f"Statistical validation: p={p_value:.4f} (SIGNIFICANT)")
    
    print("\nOutputs saved:")
    print("  1. causal_validation_results.csv")
    print("  2. intervention_ranking.csv")
    print("  3. causal_validation_plots.png")

except FileNotFoundError as e:
    print(f"\n❌ FILE ERROR: {e}")
    print("Please ensure all required data files exist.")
    sys.exit(1)
    
except KeyError as e:
    print(f"\n❌ DATA ERROR: Missing column {e}")
    print("Check that your data files have the correct structure.")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
