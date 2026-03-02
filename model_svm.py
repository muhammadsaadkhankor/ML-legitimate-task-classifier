"""
Phase 2: Support Vector Machine (SVM)
Binary Classification - Fake Task Detection
"""

from sklearn.svm import SVC
from utils import load_data, apply_borderline_smote, evaluate_model, print_results_table


def main():
    """Train and evaluate SVM with RAW and SMOTE settings"""
    print("\n" + "="*60)
    print("SUPPORT VECTOR MACHINE (SVM)")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    results = []
    
    # ========== RAW SETTING ==========
    print("\n" + "="*60)
    print("SETTING A: RAW (No Oversampling)")
    print("="*60)
    
    svm_raw = SVC(kernel='rbf', random_state=42)
    print("\nTraining SVM on RAW data...")
    svm_raw.fit(X_train, y_train)
    print("✓ Training completed")
    
    result_raw = evaluate_model(svm_raw, X_test, y_test, 'svm', 'RAW')
    results.append(result_raw)
    
    # ========== SMOTE SETTING ==========
    print("\n" + "="*60)
    print("SETTING B: BorderlineSMOTE")
    print("="*60)
    
    X_train_smote, y_train_smote = apply_borderline_smote(X_train, y_train)
    
    svm_smote = SVC(kernel='rbf', random_state=42)
    print("\nTraining SVM on SMOTE data...")
    svm_smote.fit(X_train_smote, y_train_smote)
    print("✓ Training completed")
    
    result_smote = evaluate_model(svm_smote, X_test, y_test, 'svm', 'SMOTE')
    results.append(result_smote)
    
    # ========== FINAL COMPARISON ==========
    print_results_table(results)
    
    print("\n" + "="*60)
    print("SVM COMPLETED")
    print("="*60)
    print("\nGenerated Files:")
    print("  • Data/cm_svm_raw.png")
    print("  • Data/cm_svm_smote.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
