import pandas as pd
import numpy as np

# Load the flat table
df = pd.read_csv('D:/VoiceWeave_MPDD/data/mpdd_flat.csv')

print(f"Loaded {len(df)} utterances")
print(f"\nColumns: {df.columns.tolist()}")

# Define suppression heuristic based on MPDD's emotion, position, and interaction labels
def label_suppression(row):
    """
    Label each turn as 'suppressed' or 'amplified' based on:
    - Speaker position (inferior = marginalized)
    - Emotion (negative/neutral = disengaged)
    - Number of listeners (low = isolated)
    - Next speaker's emotion (negative = unacknowledged)
    """
    
    # Suppression indicators
    is_inferior = row['speaker_position'] == 'inferior'
    negative_emotion = row['emotion'] in ['fear', 'sadness', 'disgust', 'angry', 'neutral']
    few_listeners = row['num_listeners'] <= 1
    next_negative = row['next_emotion'] in ['disgust', 'angry', 'neutral', None]
    
    # Amplification indicators
    is_peer_or_superior = row['speaker_position'] in ['peer', 'superior']
    positive_emotion = row['emotion'] in ['happiness', 'surprise']
    many_listeners = row['num_listeners'] >= 2
    next_positive = row['next_emotion'] in ['happiness', 'surprise']
    
    # Scoring system (0-4 scale for suppression, higher = more suppressed)
    suppression_score = 0
    if is_inferior:
        suppression_score += 2
    if negative_emotion:
        suppression_score += 1
    if few_listeners:
        suppression_score += 1
    if next_negative:
        suppression_score += 1
    
    # Binary label (threshold at 3+)
    if suppression_score >= 3:
        return 'suppressed', suppression_score
    elif is_peer_or_superior and positive_emotion and many_listeners:
        return 'amplified', 0
    else:
        return 'neutral', suppression_score

# Apply labeling function
df[['voice_status', 'suppression_score']] = df.apply(
    lambda row: pd.Series(label_suppression(row)), axis=1
)

# Statistics
print("\n=== Voice Status Distribution ===")
print(df['voice_status'].value_counts())
print(f"\nSuppression rate: {(df['voice_status'] == 'suppressed').sum() / len(df) * 100:.2f}%")
print(f"Amplification rate: {(df['voice_status'] == 'amplified').sum() / len(df) * 100:.2f}%")

# Check suppression by speaker position
print("\n=== Suppression by Speaker Position ===")
print(pd.crosstab(df['speaker_position'], df['voice_status'], normalize='index') * 100)

# Check suppression by emotion
print("\n=== Suppression by Emotion ===")
print(pd.crosstab(df['emotion'], df['voice_status'], normalize='index') * 100)

# Save labeled dataset
output_path = 'D:/VoiceWeave_MPDD/data/mpdd_labeled.csv'
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✅ Saved labeled dataset: {output_path}")

# Sample examples for validation
print("\n=== Sample Suppressed Turns ===")
suppressed_sample = df[df['voice_status'] == 'suppressed'].sample(3, random_state=42)
for idx, row in suppressed_sample.iterrows():
    print(f"\nDialogue {row['dialogue_id']}, Turn {row['turn_index']}")
    print(f"  Speaker: {row['speaker']} (Position: {row['speaker_position']})")
    print(f"  Emotion: {row['emotion']}")
    print(f"  Listeners: {row['num_listeners']} ({row['listeners']})")
    print(f"  Next emotion: {row['next_emotion']}")
    print(f"  Suppression score: {row['suppression_score']}")

print("\n=== Sample Amplified Turns ===")
amplified_sample = df[df['voice_status'] == 'amplified'].sample(3, random_state=42)
for idx, row in amplified_sample.iterrows():
    print(f"\nDialogue {row['dialogue_id']}, Turn {row['turn_index']}")
    print(f"  Speaker: {row['speaker']} (Position: {row['speaker_position']})")
    print(f"  Emotion: {row['emotion']}")
    print(f"  Listeners: {row['num_listeners']} ({row['listeners']})")
    print(f"  Next emotion: {row['next_emotion']}")
