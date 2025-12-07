import os
import pandas as pd
import re

DATA_DIR = "D:/VoiceWeave_MPDD/data/DailyDialog"
input_path = os.path.join(DATA_DIR, "dailydialog_clean.csv")

df = pd.read_csv(input_path)

EMOTION_MAP = {
    0: 'neutral',
    1: 'happiness',
    2: 'sadness',
    3: 'anger',
    4: 'fear',
    5: 'surprise',
    6: 'disgust'
}

utter_rows = []
for idx, row in df.iterrows():
    # Parse dialogues like "['hi.', 'hello.']"
    try:
        # Remove brackets if present, split by comma outside brackets/quotes
        raw_dialog = re.sub(r'^\[|\]$', '', str(row['dialog']).strip())
        utterances = [u.strip(" '\"\n\r") for u in raw_dialog.split("',") if u.strip(" '\"\n\r")]

        # Parse emotions: "[0 0 1]" or "[0, 0, 1]"
        emotion_str = str(row['emotion'])
        emotions = [int(e) for e in re.findall(r'\d+', emotion_str)]

        for i, utt in enumerate(utterances):
            if not utt or i >= len(emotions):
                continue
            utter_rows.append({
                'dialogue_id': idx,
                'turn_index': i,
                'speaker': f"Speaker_{(i % 2) + 1}",
                'utterance': utt,
                'emotion': EMOTION_MAP.get(emotions[i], 'neutral')
            })
    except Exception as e:
        continue

out_df = pd.DataFrame(utter_rows)
print(f"Expanded to utterance-level: {len(out_df)} utterances from {len(df)} dialogues.")
print("\nFirst 5 utterances:")
print(out_df.head())

out_path = os.path.join(DATA_DIR, "dailydialog_utterances.csv")
out_df.to_csv(out_path, index=False, encoding='utf-8')
print(f"\nSaved utterance-level table: {out_path}")
