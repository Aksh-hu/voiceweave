"""
Enhanced Predictive LSTM with Rich Sequential Features
Addresses class imbalance, adds contextual features, uses attention mechanism
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
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
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
except ImportError:
    print("❌ ERROR: PyTorch not installed!")
    print("Install with: pip install torch")
    sys.exit(1)

plt.switch_backend('Agg')

print("="*80)
print("ENHANCED PREDICTIVE LSTM: Rich Sequential Features + Class Balancing")
print("="*80)

# Custom Dataset
class DialogueDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Enhanced LSTM with Attention
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.4):
        super(AttentionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Fully connected layers
        x = self.dropout(context)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

def create_enhanced_sequences(df, sequence_length=4):
    """Create sequences with rich contextual features."""
    sequences = []
    labels = []
    
    # Pre-compute speaker-level statistics
    speaker_stats = df.groupby('speaker').agg({
        'voice_status': lambda x: (x == 'suppressed').mean()
    }).to_dict()['voice_status']
    
    for dialogue_id in df['dialogue_id'].unique():
        dialogue_data = df[df['dialogue_id'] == dialogue_id].sort_values('turn_index')
        
        if len(dialogue_data) < sequence_length + 1:
            continue
        
        for i in range(sequence_length, len(dialogue_data)):
            # Get previous turns
            prev_turns = dialogue_data.iloc[i-sequence_length:i]
            current_turn = dialogue_data.iloc[i]
            
            # Build feature vector for each turn in sequence
            seq_features = []
            for _, turn in prev_turns.iterrows():
                speaker = turn['speaker']
                
                features = [
                    turn['turn_index'],
                    turn['utterance_len'],
                    turn['word_count'],
                    turn['speaker_encoded'],
                    speaker_stats.get(speaker, 0.5),  # Speaker suppression history
                    1 if turn['voice_status'] == 'suppressed' else 0,  # Was this turn suppressed?
                    1 if turn['voice_status'] == 'amplified' else 0,   # Was this turn amplified?
                ]
                seq_features.append(features)
            
            sequences.append(seq_features)
            labels.append(1 if current_turn['voice_status'] == 'suppressed' else 0)
    
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
    
    # Feature engineering
    print("\n=== Feature Engineering ===")
    df['utterance'] = df['utterance'].fillna('')
    df['utterance_len'] = df['utterance'].str.len()
    df['word_count'] = df['utterance'].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    speaker_enc = LabelEncoder()
    df['speaker'] = df['speaker'].fillna('Unknown')
    df['speaker_encoded'] = speaker_enc.fit_transform(df['speaker'].astype(str))
    df['turn_index'] = df['turn_index'].fillna(0).astype(int)
    
    # Only use suppressed/amplified
    df = df[df['voice_status'].isin(['suppressed', 'amplified'])].copy()
    
    print(f"✓ Features ready")
    print(f"  Total examples: {len(df):,}")
    print(f"  Suppressed: {(df['voice_status'] == 'suppressed').sum():,} ({(df['voice_status'] == 'suppressed').mean():.1%})")
    print(f"  Amplified: {(df['voice_status'] == 'amplified').sum():,} ({(df['voice_status'] == 'amplified').mean():.1%})")
    
    # Create enhanced sequences
    print("\n=== Creating Enhanced Sequences ===")
    SEQUENCE_LENGTH = 4  # Use 4 previous turns
    
    X_seq, y_seq = create_enhanced_sequences(df, sequence_length=SEQUENCE_LENGTH)
    
    if len(X_seq) == 0:
        print("❌ ERROR: No sequences created!")
        sys.exit(1)
    
    print(f"✓ Generated {len(X_seq):,} sequences")
    print(f"  Shape: {X_seq.shape} (samples, timesteps, features)")
    print(f"  Suppression rate in sequences: {y_seq.mean():.2%}")
    
    # Scale features
    print("\n=== Scaling Features ===")
    scaler = StandardScaler()
    num_features = X_seq.shape[2]
    X_seq_flat = X_seq.reshape(-1, num_features)
    X_seq_scaled = scaler.fit_transform(X_seq_flat).reshape(X_seq.shape)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq_scaled, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    
    print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Compute class weights for imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weights = torch.FloatTensor(class_weights)
    
    print(f"\n=== Class Weights (for imbalance) ===")
    print(f"  Amplified (0): {class_weights[0]:.3f}")
    print(f"  Suppressed (1): {class_weights[1]:.3f}")
    
    # Create datasets
    train_dataset = DialogueDataset(X_train, y_train)
    test_dataset = DialogueDataset(X_test, y_test)
    
    # Weighted sampler for training (balance classes)
    train_targets = y_train
    sample_weights = [class_weights[int(t)] for t in train_targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Build model
    print("\n=== Building Enhanced Attention LSTM ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AttentionLSTM(input_size=num_features).to(device)
    
    # Weighted loss for class imbalance
    pos_weight = class_weights[1] / class_weights[0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    print(model)
    
    # Training
    print("\n=== Training with Class Balancing ===")
    num_epochs = 30
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_aucs = []
    best_auc = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(correct / total)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_losses.append(val_loss / len(test_loader))
        val_accs.append(correct / total)
        
        # Calculate AUC
        val_auc = roc_auc_score(all_labels, all_probs)
        val_aucs.append(val_auc)
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'D:/VoiceWeave_MPDD/results/lstm_enhanced_best.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.3f} | "
                  f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.3f}, Val AUC: {val_auc:.3f}")
    
    print(f"\n✓ Best validation AUC: {best_auc:.3f}")
    
    # Load best model
    model.load_state_dict(torch.load('D:/VoiceWeave_MPDD/results/lstm_enhanced_best.pth'))
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            predicted = (probs > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_pred_prob = np.array(all_probs)
    
    test_acc = (y_pred == y_test).mean()
    test_auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Test AUC-ROC: {test_auc:.3f}")
    
    print("\n" + classification_report(y_test, y_pred, target_names=['Amplified', 'Suppressed']))
    
    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_prob': y_pred_prob,
        'predicted_label': y_pred
    })
    pred_df.to_csv('D:/VoiceWeave_MPDD/results/lstm_enhanced_predictions.csv', index=False)
    
    # Visualization
    print("\n=== Creating Plots ===")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Training curves
    axes[0, 0].plot(train_accs, label='Train', linewidth=2, color='blue')
    axes[0, 0].plot(val_accs, label='Validation', linewidth=2, color='orange')
    axes[0, 0].set_title('Accuracy Over Epochs', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(val_aucs, label='Validation AUC', linewidth=2, color='green')
    axes[0, 1].axhline(0.5, color='red', linestyle='--', label='Random Baseline', linewidth=1)
    axes[0, 1].set_title('AUC-ROC Over Epochs', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('AUC-ROC', fontsize=11)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # Prediction distribution
    axes[1, 0].hist(y_pred_prob[y_test == 0], bins=30, alpha=0.6, label='Amplified (True)', color='green', edgecolor='black')
    axes[1, 0].hist(y_pred_prob[y_test == 1], bins=30, alpha=0.6, label='Suppressed (True)', color='red', edgecolor='black')
    axes[1, 0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 0].set_title('Prediction Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Probability', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=['Amplified', 'Suppressed'],
                yticklabels=['Amplified', 'Suppressed'],
                annot_kws={"size": 14})
    axes[1, 1].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Predicted', fontsize=11)
    axes[1, 1].set_ylabel('True', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('D:/VoiceWeave_MPDD/results/lstm_enhanced_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Plots saved")
    
    print("\n" + "="*80)
    print("✅ ENHANCED PREDICTIVE LSTM COMPLETE!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  Test AUC: {test_auc:.2%} (Target: >70%)")
    print(f"  Test Accuracy: {test_acc:.2%}")
    print(f"  Best validation AUC: {best_auc:.2%}")
    
    if test_auc > 0.70:
        print("\n🎯 STRONG RESULT: AUC > 70% achieved!")
    elif test_auc > 0.65:
        print("\n✅ GOOD RESULT: Meaningful predictive power demonstrated")
    else:
        print("\n⚠️  Moderate result. Consider more features or data.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
