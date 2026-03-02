"""
Utility Functions for Phase 2: Model Training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


def load_data(train_path='Data/train_preprocessed.csv', test_path='Data/test_preprocessed.csv'):
    """Load preprocessed train and test data"""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['Legitimacy'])
    y_train = train_df['Legitimacy']
    X_test = test_df.drop(columns=['Legitimacy'])
    y_test = test_df['Legitimacy']
    
    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Train class: Fake={sum(y_train==0)} ({sum(y_train==0)/len(y_train)*100:.2f}%), "
          f"Legit={sum(y_train==1)} ({sum(y_train==1)/len(y_train)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test


def apply_borderline_smote(X_train, y_train, random_state=42):
    """Apply BorderlineSMOTE to training data only"""
    print("\nApplying BorderlineSMOTE...")
    smote = BorderlineSMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_resampled.shape[0]} samples")
    print(f"Class dist: Fake={sum(y_resampled==0)}, Legit={sum(y_resampled==1)}")
    return X_resampled, y_resampled


def compute_mutual_information(TP, FN, FP, TN):
    """Calculate MI using I(X;Y) = H(X) - H(X|Y) with log2 (bits)"""
    Total = TP + FN + FP + TN
    
    # Prior probabilities
    P_fake = (TP + FN) / Total
    P_legit = (FP + TN) / Total
    
    # Initial entropy H(X)
    H_X = 0
    if P_fake > 0:
        H_X -= P_fake * np.log2(P_fake)
    if P_legit > 0:
        H_X -= P_legit * np.log2(P_legit)
    
    # Marginals for predictions
    P_pred_fake = (TP + FP) / Total
    P_pred_legit = (FN + TN) / Total
    
    # Posteriors
    P_fake_given_pred_fake = TP / (TP + FP) if (TP + FP) > 0 else 0
    P_fake_given_pred_legit = FN / (FN + TN) if (FN + TN) > 0 else 0
    
    # Binary entropy helper
    def binary_entropy(p):
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    # Conditional entropies
    H_given_pred_fake = binary_entropy(P_fake_given_pred_fake)
    H_given_pred_legit = binary_entropy(P_fake_given_pred_legit)
    
    # Average conditional entropy H(X|Y)
    H_XY = P_pred_fake * H_given_pred_fake + P_pred_legit * H_given_pred_legit
    
    # Mutual Information
    MI = H_X - H_XY
    
    return MI, H_X, H_XY


def evaluate_model(model, X_test, y_test, model_name, setting):
    """Evaluate model and return results"""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name} - {setting}")
    print(f"{'='*60}")
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Mutual Information
    mi, H_X, H_XY = compute_mutual_information(TP, FN, FP, TN)
    
    # Normalized weights
    weights = cm / cm.sum()
    weights_normalized = weights / weights.sum()
    
    # Print results
    print(f"\n{setting} | {model_name}")
    print(f"  Acc={acc:.4f} | "
          f"Fake: P={precision_per_class[0]:.4f} R={recall_per_class[0]:.4f} F1={f1_per_class[0]:.4f} | "
          f"Legit: P={precision_per_class[1]:.4f} R={recall_per_class[1]:.4f} F1={f1_per_class[1]:.4f} | "
          f"MacroF1={f1_macro:.4f}")
    
    print(f"\nMutual Information: {mi:.6f}")
    print(f"Normalized Weights:")
    print(f"  [[{weights_normalized[0,0]:.4f}, {weights_normalized[0,1]:.4f}],")
    print(f"   [{weights_normalized[1,0]:.4f}, {weights_normalized[1,1]:.4f}]]")
    
    # Save confusion matrix
    cm_filename = f"Data/cm_{model_name.lower().replace(' ', '_')}_{setting.lower()}.png"
    plot_confusion_matrix(cm, model_name, setting, cm_filename)
    print(f"\n✓ Saved: {cm_filename}")
    
    return {
        'setting': setting,
        'model': model_name,
        'accuracy': acc,
        'precision_fake': precision_per_class[0],
        'recall_fake': recall_per_class[0],
        'f1_fake': f1_per_class[0],
        'precision_legit': precision_per_class[1],
        'recall_legit': recall_per_class[1],
        'f1_legit': f1_per_class[1],
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'mutual_information': mi,
        'normalized_weights': weights_normalized
    }


def plot_confusion_matrix(cm, model_name, setting, save_path):
    """Plot and save confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Pred Fake', 'Pred Legit'],
                yticklabels=['True Fake', 'True Legit'])
    plt.title(f'Confusion Matrix: {model_name} ({setting})', fontweight='bold', fontsize=14)
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_results_table(results):
    """Print results in formatted table"""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for result in results:
        setting = result['setting']
        model = result['model']
        print(f"\n{setting} | {model}")
        print(f"  Acc={result['accuracy']:.4f} | "
              f"Fake: P={result['precision_fake']:.4f} R={result['recall_fake']:.4f} F1={result['f1_fake']:.4f} | "
              f"Legit: P={result['precision_legit']:.4f} R={result['recall_legit']:.4f} F1={result['f1_legit']:.4f} | "
              f"MacroF1={result['f1_macro']:.4f}")
        print(f"  MI={result['mutual_information']:.6f}")
