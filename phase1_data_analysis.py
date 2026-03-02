"""
Phase 1: Dataset Loading and Analysis
Binary Classification - Fake Task Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_dataset(filepath):
    """Load dataset from CSV file"""
    df = pd.read_csv(filepath)
    print(f"✓ Loaded: {filepath} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df


def display_basic_info(df):
    """Display basic dataset information"""
    print("\n" + "="*60)
    print("BASIC INFORMATION")
    print("="*60)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nData Types:\n{df.dtypes}")
    
    missing = df.isnull().sum()
    print(f"\nMissing Values: {missing.sum()}")
    if missing.sum() > 0:
        print(missing[missing > 0])
    
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicates}")


def numerical_statistics(df):
    """Display numerical feature statistics"""
    print("\n" + "="*60)
    print("NUMERICAL STATISTICS")
    print("="*60)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = df[numerical_cols].describe().T
    stats['variance'] = df[numerical_cols].var()
    print(stats[['mean', 'std', 'min', 'max', 'variance']].round(2))
    return numerical_cols


def analyze_target_distribution(df, target_col='Legitimacy'):
    """Analyze and visualize target column distribution"""
    print("\n" + "="*60)
    print("TARGET DISTRIBUTION")
    print("="*60)
    counts = df[target_col].value_counts().sort_index()
    percentages = (df[target_col].value_counts(normalize=True).sort_index() * 100)
    
    for cls in counts.index:
        label = "Legitimate" if cls == 1 else "Fake"
        print(f"Class {cls} ({label}): {counts[cls]:,} ({percentages[cls]:.2f}%)")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(counts.index, counts.values, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Class', fontweight='bold')
    axes[0].set_ylabel('Count', fontweight='bold')
    axes[0].set_title('Class Distribution (Count)', fontweight='bold')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Fake (0)', 'Legitimate (1)'])
    for idx, val in counts.items():
        axes[0].text(idx, val, f'{val:,}', ha='center', va='bottom', fontweight='bold')
    
    axes[1].pie(percentages.values, labels=['Fake (0)', 'Legitimate (1)'], 
                autopct='%1.2f%%', colors=['#e74c3c', '#2ecc71'], startangle=90, explode=(0.05, 0))
    axes[1].set_title('Class Distribution (%)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Data/class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Data/class_distribution.png")
    plt.close()


def plot_feature_histograms(df, numerical_cols, target_col='Legitimacy'):
    """Plot histograms for all numerical features"""
    print("\n" + "="*60)
    print("FEATURE HISTOGRAMS")
    print("="*60)
    feature_cols = [col for col in numerical_cols if col != target_col]
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        axes[idx].hist(df[col], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel(col, fontweight='bold')
        axes[idx].set_ylabel('Frequency', fontweight='bold')
        axes[idx].set_title(f'{col}', fontweight='bold')
    
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('Data/feature_histograms.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Data/feature_histograms.png")
    plt.close()


def plot_correlation_heatmap(df, numerical_cols):
    """Plot correlation heatmap"""
    print("\n" + "="*60)
    print("CORRELATION HEATMAP")
    print("="*60)
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('Data/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Data/correlation_heatmap.png")
    plt.close()
    
    print("\nHighly Correlated Pairs (|r| > 0.7):")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr:
        for feat1, feat2, corr_val in high_corr:
            print(f"  {feat1} ↔ {feat2}: {corr_val:.3f}")
    else:
        print("  None found")


def check_outliers_boxplots(df, numerical_cols, target_col='Legitimacy'):
    """Check for outliers using boxplots"""
    print("\n" + "="*60)
    print("OUTLIER BOXPLOTS")
    print("="*60)
    feature_cols = [col for col in numerical_cols if col != target_col]
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        axes[idx].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))
        axes[idx].set_ylabel(col, fontweight='bold')
        axes[idx].set_title(f'{col}', fontweight='bold')
    
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('Data/outlier_boxplots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Data/outlier_boxplots.png")
    plt.close()


def preprocess_data(df, target_col='Legitimacy', scaler=None):
    """Preprocess dataset - fit scaler if None, else use provided scaler"""
    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)
    
    # Drop duplicates
    original_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates removed: {original_rows - len(df)}")
    
    # Handle missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Missing values: {missing}")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
    else:
        print("Missing values: 0")
    
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        for col in categorical_cols:
            df[col] = pd.Categorical(df[col]).codes
        print(f"Encoded categorical: {categorical_cols}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"✓ Fitted StandardScaler on {X.shape[1]} features")
    else:
        X_scaled = scaler.transform(X)
        print(f"✓ Transformed using fitted scaler")
    
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("PHASE 1: DATASET LOADING AND ANALYSIS")
    print("="*60)
    
    # ========== TRAIN DATA ==========
    print("\n" + "="*60)
    print("PROCESSING TRAIN DATA")
    print("="*60)
    
    train_df = load_dataset('Data/train.csv')
    display_basic_info(train_df)
    numerical_cols = numerical_statistics(train_df)
    analyze_target_distribution(train_df)
    plot_feature_histograms(train_df, numerical_cols)
    plot_correlation_heatmap(train_df, numerical_cols)
    check_outliers_boxplots(train_df, numerical_cols)
    
    X_train, y_train, scaler = preprocess_data(train_df, scaler=None)
    
    train_preprocessed = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    train_preprocessed.to_csv('Data/train_preprocessed.csv', index=False)
    print(f"✓ Saved: Data/train_preprocessed.csv ({len(train_preprocessed)} rows)")
    
    # ========== TEST DATA ==========
    print("\n" + "="*60)
    print("PROCESSING TEST DATA")
    print("="*60)
    
    test_df = load_dataset('Data/test.csv')
    display_basic_info(test_df)
    
    X_test, y_test, _ = preprocess_data(test_df, scaler=scaler)
    
    test_preprocessed = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    test_preprocessed.to_csv('Data/test_preprocessed.csv', index=False)
    print(f"✓ Saved: Data/test_preprocessed.csv ({len(test_preprocessed)} rows)")
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Original train.csv: 11,587 rows")
    print(f"Preprocessed train: {len(train_preprocessed)} rows")
    print(f"Original test.csv: 2,897 rows")
    print(f"Preprocessed test: {len(test_preprocessed)} rows")
    print(f"\nFeatures: {X_train.shape[1]}")
    print(f"Train class distribution: Fake={sum(y_train==0)}, Legitimate={sum(y_train==1)}")
    print(f"Test class distribution: Fake={sum(y_test==0)}, Legitimate={sum(y_test==1)}")
    print("\n✓ No data leakage - scaler fitted only on train data")
    print("✓ All plots saved to Data/ folder")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
