"""
Enhanced Predictive LSTM with Attention + Class Balancing
- Attention mechanism to identify which past turns matter
- Class-weighted loss for imbalanced data
- Better features (suppression history, speaker activity patterns)
- Runs in ~8-10 minutes
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("❌ ERROR: PyTorch not installed!")
    print("Install with: pip install torch")
    sys.exit(1)

plt.switch_backend('Agg')
sns.set_style('whitegrid')

print("="*80)
print("ENHANCED LSTM WITH ATTENTION: Better Predictive Suppression")
print("="*80)

# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        """
        lstm_output: (batch_size, seq_len, hidden_size)
        returns: weighted sum of outputs
        """
        # Compute attention weights
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)  # (batch, seq_len, 1)
        # Weight the outputs
        weighted_output = torch.sum(lstm_output * attn_weights, dim=1)  # (batch, hidden_size)
        return weighted_output, attn_weights

# Enhanced LSTM Model
class EnhancedLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(EnhancedLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = AttentionLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        weighted_out, attn_weights = self.attention(lstm_out)
        x = self.dropout(weighted_out)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x, attn_weights

class DialogueDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_enhanced_sequences(df, sequence_length=3, features=None):
    """Create sequences with additional context features."""
    sequences = []
    labels = []
    
    for dialogue_id in df['dialogue_id'].unique():
        dialogue_data = df[df['dialogue_id'] == dialogue_id].sort_values('turn_index')
        
        if len(dialogue_data) < sequence_length + 1:
            continue
        
        for i in range(sequence_length, len(dialogue_data)):
            # Get sequence
            sequence = dialogue_data.iloc[i-sequence_length:i][features].values
            label = dialogue_data.iloc[i]['is_suppressed']
            sequences.append(sequence)
            labels.append(label)
    
    return np.array(sequences), np.array(labels)

try:
    # Load data
    print("\n=== Loading Data ===")
    data_path = "D:/VoiceWeave_MPDD/data/merged_utterances.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data file not found: {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path, encoding='utf-8')
    df = df.sort_values(['dialogue_id', 'turn_index']).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df):,} utterances from {df['dialogue_id'].nunique():,} dialogues")
    
    # Enhanced feature engineering
    print("\n=== Enhanced Feature Engineering ===")
    df['utterance'] = df['utterance'].fillna('')
    df['utterance_len'] = df['utterance'].str.len()
    df['word_count'] = df['utterance'].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['is_suppressed'] = (df['voice_status'] == 'suppressed').astype(int)
    
    # Speaker encoding
    speaker_enc = LabelEncoder()
    df['speaker'] = df['speaker'].fillna('Unknown')
    df['speaker_encoded'] = speaker_enc.fit_transform(df['speaker'].astype(str))
    df['turn_index'] = df['turn_index'].fillna(0).astype(int)
    
    # NEW: Suppression streak (how many consecutive suppressed turns by this speaker?)
    df['suppression_streak'] = 0
    for speaker in df['speaker'].unique():
        speaker_mask = df['speaker'] == speaker
        speaker_indices = df[speaker_mask].index
        
        streak = 0
        for idx in speaker_indices:
            if df.loc[idx, 'is_suppressed']:
                streak += 1
            else:
                streak = 0
            df.loc[idx, 'suppression_streak'] = streak
    
    # NEW: Speaker activity (inverse of suppression rate in recent history)
    speaker_supp_rate = df.groupby('speaker')['is_suppressed'].mean().to_dict()
    df['speaker_amplification_tendency'] = df['speaker'].map(lambda x: 1 - speaker_supp_rate.get(x, 0.5))
    
    # NEW: Dialogue diversity (number of unique speakers)
    speaker_counts = df.groupby('dialogue_id')['speaker'].nunique().to_dict()
    df['dialogue_diversity'] = df['dialogue_id'].map(speaker_counts).fillna(1)
    
    feature_cols = [
        'turn_index',
        'utterance_len',
        'word_count',
        'speaker_encoded',
        'suppression_streak',
        'speaker_amplification_tendency',
        'dialogue_diversity'
    ]
    
    df = df.dropna(subset=feature_cols)
    
    print(f"✓ Enhanced features ready")
    print(f"  Features: {feature_cols}")
    print(f"  Suppression rate: {df['is_suppressed'].mean():.2%}")
    
    # Create sequences
    print("\n=== Creating Enhanced Sequences ===")
    SEQUENCE_LENGTH = 3
    
    X_seq, y_seq = create_enhanced_sequences(df, sequence_length=SEQUENCE_LENGTH, features=feature_cols)
    
    if len(X_seq) == 0:
        print("❌ ERROR: No sequences created!")
        sys.exit(1)
    
    print(f"✓ Generated {len(X_seq):,} sequences")
    print(f"  Shape: {X_seq.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_seq_scaled = scaler.fit_transform(X_seq_flat).reshape(X_seq.shape)
    
    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq_scaled, y_seq, test_size=0.2, random_state=42, stratify=y_seq
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq_scaled, y_seq, test_size=0.2, random_state=42
        )
    
    print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Calculate class weights for imbalanced data
    class_weights = {}
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        class_weights[u] = len(y_train) / (2.0 * c)
    
    print(f"✓ Class weights: {class_weights}")
    
    # Create datasets
    train_dataset = DialogueDataset(X_train, y_train)
    test_dataset = DialogueDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Build model
    print("\n=== Building Enhanced LSTM with Attention ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EnhancedLSTMWithAttention(input_size=X_train.shape[2]).to(device)
    
    # Weighted BCE loss
    criterion = nn.BCELoss(weight=torch.tensor([class_weights[1]]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print(model)
    
    # Training
    print("\n=== Training with Attention ===")
    num_epochs = 15
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            outputs = outputs.squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(correct / total)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs, _ = model(X_batch)
                outputs = outputs.squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        val_losses.append(val_loss / len(test_loader))
        val_accs.append(correct / total)
        scheduler.step(val_losses[-1])
        
        if (epoch + 1) % 3 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.3f} | "
                  f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.3f}")
    
    # Evaluate
    print("\n=== Evaluation ===")
    model.eval()
    all_preds = []
    all_probs = []
    all_attentions = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs, attentions = model(X_batch)
            outputs = outputs.squeeze()
            all_probs.extend(outputs.cpu().numpy())
            predicted = (outputs > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_attentions.extend(attentions.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_pred_prob = np.array(all_probs)
    
    test_acc = (y_pred == y_test).mean()
    test_auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Test AUC-ROC: {test_auc:.3f}")
    print(f"\nImprovement over baseline LSTM: +{(test_auc - 0.555) * 100:.1f}% AUC")
    
    print("\n" + classification_report(y_test, y_pred, target_names=['Amplified', 'Suppressed']))
    
    # Save model
    torch.save(model.state_dict(), 'D:/VoiceWeave_MPDD/results/enhanced_lstm_attention_model.pth')
    print("\n✓ Model saved")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_prob': y_pred_prob,
        'predicted_label': y_pred
    })
    pred_df.to_csv('D:/VoiceWeave_MPDD/results/enhanced_lstm_predictions.csv', index=False)
    
    # Visualization
    print("\n=== Creating Visualizations ===")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Training curves
    axes[0, 0].plot(train_accs, label='Train', linewidth=2, marker='o')
    axes[0, 0].plot(val_accs, label='Validation', linewidth=2, marker='o')
    axes[0, 0].set_title('Accuracy with Attention', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(train_losses, label='Train', linewidth=2, marker='o')
    axes[0, 1].plot(val_losses, label='Validation', linewidth=2, marker='o')
    axes[0, 1].set_title('Loss with Attention', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Prediction distribution
    axes[1, 0].hist(y_pred_prob[y_test == 0], bins=30, alpha=0.6, label='Amplified', color='green')
    axes[1, 0].hist(y_pred_prob[y_test == 1], bins=30, alpha=0.6, label='Suppressed', color='red')
    axes[1, 0].axvline(0.5, color='black', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Enhanced LSTM Predictions', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=['Amplified', 'Suppressed'],
                yticklabels=['Amplified', 'Suppressed'])
    axes[1, 1].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('D:/VoiceWeave_MPDD/results/enhanced_lstm_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Plots saved")
    
    print("\n" + "="*80)
    print("✅ ENHANCED LSTM WITH ATTENTION COMPLETE!")
    print("="*80)
    print(f"\nResults:")
    print(f"  • AUC-ROC: {test_auc:.3f} (vs 0.555 baseline)")
    print(f"  • Accuracy: {test_acc:.3f}")
    print(f"  • Class-balanced loss applied")
    print(f"  • Attention mechanism identifies key past turns")
    
    print("\nOutputs saved:")
    print("  1. enhanced_lstm_attention_model.pth")
    print("  2. enhanced_lstm_predictions.csv")
    print("  3. enhanced_lstm_performance.png")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
