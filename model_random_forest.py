"""
Phase 2: Random Forest Classifier
Binary Classification - Fake Task Detection
"""

from sklearn.ensemble import RandomForestClassifier
from utils import load_data, apply_borderline_smote, evaluate_model, print_results_table


def main():
    """Train and evaluate Random Forest with RAW and SMOTE settings"""
    print("\n" + "="*60)
    print("RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    results = []
    
    # ========== RAW SETTING ==========
    print("\n" + "="*60)
    print("SETTING A: RAW (No Oversampling)")
    print("="*60)
    
    rf_raw = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("\nTraining Random Forest on RAW data...")
    rf_raw.fit(X_train, y_train)
    print("✓ Training completed")
    
    result_raw = evaluate_model(rf_raw, X_test, y_test, 'rf', 'RAW')
    results.append(result_raw)
    
    # ========== SMOTE SETTING ==========
    print("\n" + "="*60)
    print("SETTING B: BorderlineSMOTE")
    print("="*60)
    
    X_train_smote, y_train_smote = apply_borderline_smote(X_train, y_train)
    
    rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("\nTraining Random Forest on SMOTE data...")
    rf_smote.fit(X_train_smote, y_train_smote)
    print("✓ Training completed")
    
    result_smote = evaluate_model(rf_smote, X_test, y_test, 'rf', 'SMOTE')
    results.append(result_smote)
    
    # ========== FINAL COMPARISON ==========
    print_results_table(results)
    
    print("\n" + "="*60)
    print("RANDOM FOREST COMPLETED")
    print("="*60)
    print("\nGenerated Files:")
    print("  • Data/cm_rf_raw.png")
    print("  • Data/cm_rf_smote.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
