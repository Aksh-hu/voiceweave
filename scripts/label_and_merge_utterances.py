import pandas as pd

# Paths
mpdd_file = "D:/VoiceWeave_MPDD/data/mpdd_flat.csv"  # Original flat MPDD as used earlier
dd_file = "D:/VoiceWeave_MPDD/data/DailyDialog/dailydialog_utterances.csv"

mpdd_df = pd.read_csv(mpdd_file)
dd_df   = pd.read_csv(dd_file)

# Add dataset source
mpdd_df['source'] = 'MPDD'
dd_df['source'] = 'DailyDialog'

# Proxy suppression labeling: same logic for both
def label_dd(row):
    suppressed = row['emotion'] in ['fear', 'sadness', 'neutral', 'disgust', 'anger']
    amplified = row['emotion'] in ['happiness', 'surprise']
    if amplified:
        return 'amplified'
    elif suppressed:
        return 'suppressed'
    else:
        return 'neutral'

for df in [mpdd_df, dd_df]:
    df['voice_status'] = df.apply(label_dd, axis=1)

# Select/align columns: ['dialogue_id', 'turn_index', 'speaker', 'utterance', 'emotion', 'voice_status', 'source']
# For MPDD, keep only these
mpdd_reduced = mpdd_df[['dialogue_id', 'turn_index', 'speaker', 'utterance', 'emotion', 'voice_status', 'source']]
dd_reduced = dd_df[['dialogue_id', 'turn_index', 'speaker', 'utterance', 'emotion', 'voice_status', 'source']]

# Merge datasets
full_df = pd.concat([mpdd_reduced, dd_reduced], ignore_index=True)
full_df.to_csv("D:/VoiceWeave_MPDD/data/merged_utterances.csv", index=False, encoding='utf-8')
print(f"Final merged set size: {len(full_df)} utterances")
print(full_df['source'].value_counts())
print(full_df['voice_status'].value_counts())
print(full_df.sample(5, random_state=42))
