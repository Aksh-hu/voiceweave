import json
import pandas as pd

# Load the MPDD dataset (correct path from scripts folder)
with open('D:/VoiceWeave_MPDD/data/Dialogue-MPDD/mpdd/dialogue.json', 'r', encoding='utf-8') as f:
    dialogues = json.load(f)

with open('D:/VoiceWeave_MPDD/data/Dialogue-MPDD/mpdd/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Print basic stats
print(f"Total dialogues: {len(dialogues)}")
print(f"\nEmotion types: {metadata['emotion']}")
print(f"\nRelation positions: {list(metadata['position'].keys())}")
print(f"\nRelation fields: {list(metadata['field'].keys())}")

# Inspect first dialogue
first_dialogue_key = list(dialogues.keys())[0]
first_dialogue = dialogues[first_dialogue_key]

print(f"\n=== First Dialogue (ID: {first_dialogue_key}) ===")
print(f"Number of turns: {len(first_dialogue)}\n")

for i, turn in enumerate(first_dialogue[:5]):  # Show first 5 turns
    print(f"Turn {i+1}:")
    print(f"  Speaker: {turn['speaker']}")
    print(f"  Emotion: {turn['emotion']}")
    print(f"  Utterance: {turn['utterance'][:80]}...")  # First 80 chars
    print(f"  Listeners: {[l['name'] for l in turn['listener']]}")
    print(f"  Relations: {[l['relation'] for l in turn['listener']]}")
    print()

# Count total utterances
total_utterances = sum(len(d) for d in dialogues.values())
print(f"\nTotal utterances across all dialogues: {total_utterances}")
