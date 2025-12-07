import json
import pandas as pd
from collections import Counter

# Load MPDD data
with open('D:/VoiceWeave_MPDD/data/Dialogue-MPDD/mpdd/dialogue.json', 'r', encoding='utf-8') as f:
    dialogues = json.load(f)

with open('D:/VoiceWeave_MPDD/data/Dialogue-MPDD/mpdd/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Flatten all dialogues into rows
flat_data = []

for dialogue_id, turns in dialogues.items():
    for turn_idx, turn in enumerate(turns):
        # Basic turn info
        row = {
            'dialogue_id': dialogue_id,
            'turn_index': turn_idx,
            'speaker': turn['speaker'],
            'utterance': turn['utterance'],
            'emotion': turn['emotion'],
            'num_listeners': len(turn['listener']),
            'listeners': ','.join([l['name'] for l in turn['listener']]),
            'relations': ','.join([l['relation'] for l in turn['listener']]),
        }
        
        # Extract speaker position (superior/peer/inferior)
        # Check first relation to determine position
        if turn['listener']:
            first_relation = turn['listener'][0]['relation']
            for position, relations_list in metadata['position'].items():
                if first_relation in relations_list:
                    row['speaker_position'] = position
                    break
            else:
                row['speaker_position'] = 'unknown'
        else:
            row['speaker_position'] = 'no_listener'
        
        # Context: previous turn (if exists)
        if turn_idx > 0:
            prev_turn = turns[turn_idx - 1]
            row['prev_speaker'] = prev_turn['speaker']
            row['prev_emotion'] = prev_turn['emotion']
            row['prev_utterance_len'] = len(prev_turn['utterance'])
        else:
            row['prev_speaker'] = None
            row['prev_emotion'] = None
            row['prev_utterance_len'] = 0
        
        # Context: next turn (if exists)
        if turn_idx < len(turns) - 1:
            next_turn = turns[turn_idx + 1]
            row['next_speaker'] = next_turn['speaker']
            row['next_emotion'] = next_turn['emotion']
        else:
            row['next_speaker'] = None
            row['next_emotion'] = None
        
        # Utterance features
        row['utterance_len'] = len(turn['utterance'])
        row['utterance_word_count'] = len(turn['utterance'].split())
        
        flat_data.append(row)

# Create DataFrame
df = pd.DataFrame(flat_data)

# Save to CSV
output_path = 'D:/VoiceWeave_MPDD/data/mpdd_flat.csv'
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Saved flat table: {output_path}")
print(f"Total rows: {len(df)}")
print(f"\nFirst 5 rows preview:")
print(df.head())
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nEmotion distribution:")
print(df['emotion'].value_counts())
print(f"\nSpeaker position distribution:")
print(df['speaker_position'].value_counts())
