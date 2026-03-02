"""
PyTorch MLP for Binary Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
import matplotlib.pyplot as plt
import seaborn as sns

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History - Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training History - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, setting, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Fake (0)', 'Legitimate (1)'],
                yticklabels=['Fake (0)', 'Legitimate (1)'])
    plt.title(f'Confusion Matrix: PyTorch MLP ({setting})', fontweight='bold', fontsize=14)
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, setting):
    print(f"\n{'='*60}")
    print(f"PYTORCH MLP - {setting} - 100 EPOCHS")
    print(f"{'='*60}")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Convert to tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MLP(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    
    print("\nTraining for 100 epochs...")
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        y_train_true, y_train_pred = evaluate(model, train_loader, device)
        y_val_true, y_val_pred = evaluate(model, val_loader, device)
        
        train_acc = accuracy_score(y_train_true, y_train_pred)
        val_acc = accuracy_score(y_val_true, y_val_pred)
        
        # Calculate val loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100 - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'Data/best_mlp_pytorch.pth')
    
    # Load best model
    model.load_state_dict(torch.load('Data/best_mlp_pytorch.pth'))
    
    # Test evaluation
    y_test_true, y_test_pred = evaluate(model, test_loader, device)
    
    test_acc = accuracy_score(y_test_true, y_test_pred)
    precision = precision_score(y_test_true, y_test_pred, average=None, zero_division=0)
    recall = recall_score(y_test_true, y_test_pred, average=None, zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, average=None, zero_division=0)
    test_f1_weighted = f1_score(y_test_true, y_test_pred, average='weighted')
    test_f1_macro = f1_score(y_test_true, y_test_pred, average='macro')
    cm = confusion_matrix(y_test_true, y_test_pred)
    
    print("\n" + "="*60)
    print(f"FINAL RESULTS - {setting}")
    print("="*60)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"\nClass 0 (Fake):")
    print(f"  Precision: {precision[0]:.4f}")
    print(f"  Recall:    {recall[0]:.4f}")
    print(f"  F1-Score:  {f1[0]:.4f}")
    print(f"\nClass 1 (Legitimate):")
    print(f"  Precision: {precision[1]:.4f}")
    print(f"  Recall:    {recall[1]:.4f}")
    print(f"  F1-Score:  {f1[1]:.4f}")
    print(f"\nMacro F1:    {test_f1_macro:.4f}")
    print(f"Weighted F1: {test_f1_weighted:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Save plots
    setting_lower = setting.lower().replace(' ', '_')
    plot_training_history(history, f'Data/mlp_pytorch_training_history_{setting_lower}.png')
    plot_confusion_matrix(cm, setting, f'Data/cm_mlp_pytorch_{setting_lower}.png')
    torch.save(model.state_dict(), f'Data/best_mlp_pytorch_{setting_lower}.pth')
    
    print(f"\n✓ Saved: Data/mlp_pytorch_training_history_{setting_lower}.png")
    print(f"✓ Saved: Data/cm_mlp_pytorch_{setting_lower}.png")
    print(f"✓ Saved: Data/best_mlp_pytorch_{setting_lower}.pth")
    print("="*60)

def main():
    print("="*60)
    print("PYTORCH MLP - RAW & SMOTE")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv('Data/train_preprocessed.csv')
    test_df = pd.read_csv('Data/test_preprocessed.csv')
    
    X_train_full = train_df.drop(columns=['Legitimacy']).values
    y_train_full = train_df['Legitimacy'].values
    X_test = test_df.drop(columns=['Legitimacy']).values
    y_test = test_df['Legitimacy'].values
    
    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )
    
    print(f"\nOriginal - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Train class dist: Fake={sum(y_train==0)}, Legit={sum(y_train==1)}")
    
    # ========== RAW SETTING ==========
    train_model(X_train, y_train, X_val, y_val, X_test, y_test, 'RAW')
    
    # ========== SMOTE SETTING ==========
    print("\n" + "="*60)
    print("APPLYING BORDERLINE SMOTE")
    print("="*60)
    smote = BorderlineSMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Train: {X_train_smote.shape[0]}")
    print(f"Train class dist: Fake={sum(y_train_smote==0)}, Legit={sum(y_train_smote==1)}")
    
    train_model(X_train_smote, y_train_smote, X_val, y_val, X_test, y_test, 'SMOTE')

if __name__ == "__main__":
    main()
