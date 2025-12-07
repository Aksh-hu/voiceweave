import os
import pandas as pd

# Set your DailyDialog folder path
DATA_DIR = "D:/VoiceWeave_MPDD/data/DailyDialog"

# Define the filenames (update if using other names)
train_file = os.path.join(DATA_DIR, "train.csv")
valid_file = os.path.join(DATA_DIR, "validation.csv")
test_file  = os.path.join(DATA_DIR, "test.csv")

# Load datasets
train_df = pd.read_csv(train_file)
valid_df = pd.read_csv(valid_file)
test_df  = pd.read_csv(test_file)

print(f"Train examples: {len(train_df)}")
print(f"Valid examples: {len(valid_df)}")
print(f"Test examples:  {len(test_df)}")

# Combine splits
dd_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
print(f"Total combined DailyDialog rows: {len(dd_df)}")
print("\nFirst 3 rows:")
print(dd_df.head(3))

# Example column mapping for typical DailyDialog format (update these if needed!)
# Common columns: ['dialogue_id', 'speaker', 'utterance', 'emotion', ...]
# If not present, you'll need to split dialogues using '\t' or row per utterance

REQUIRED_COLUMNS = ['dialogue_id', 'speaker', 'utterance', 'emotion']

if all(c in dd_df.columns for c in REQUIRED_COLUMNS):
    print("Columns are in expected format.")
else:
    # For standard DailyDialog, you may need custom parsing
    # Example: 'dialogue' column contains all utterances; 'emotion' column has emotion for each utterance
    if 'dialogue' in dd_df.columns and 'emotion' in dd_df.columns:
        def extract_utterances(row):
            utterances = row['dialogue'].split('__eou__')
            emotions = str(row['emotion']).split()
            items = []
            for i, utt in enumerate(utterances):
                utt = utt.strip()
                if utt:
                    items.append({
                        'dialogue_id': row.name,
                        'speaker': f'Speaker_{i % 2 + 1}',  # Alternating speakers if unknown
                        'utterance': utt,
                        'emotion': emotions[i] if i < len(emotions) else 'neutral',
                    })
            return items

        dd_rows = []
        for idx, row in dd_df.iterrows():
            dd_rows.extend(extract_utterances(row))

        dd_df = pd.DataFrame(dd_rows)
        print(f"Expanded to utterance-level: {len(dd_df)} turns.")
        print(dd_df.head(5))

# Save unified DailyDialog data for downstream work
output_path = os.path.join(DATA_DIR, "dailydialog_clean.csv")
dd_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\nSaved unified DailyDialog utterance table: {output_path}")
print("\nSample 5 utterances:")
print(dd_df.sample(5, random_state=42))
