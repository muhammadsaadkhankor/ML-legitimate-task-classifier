"""
Phase 3: Results Summary and Analysis
Binary Classification - Fake Task Detection
Using Professor's MI Formula: I(X;Y) = H(X) - H(X|Y) with log2 (bits)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Confusion matrices from Phase 2 results
# Format: (TP, FN, FP, TN)
confusion_matrices = {
    'RAW': {
        'RandomForest': (365, 14, 0, 2518),
        'SVM': (82, 297, 33, 2485),
        'MLP': (360, 19, 9, 2509)
    },
    'SMOTE': {
        'RandomForest': (364, 15, 8, 2510),
        'SVM': (336, 43, 500, 2018),
        'MLP': (365, 14, 16, 2502)
    }
}


def compute_mutual_information(TP, FN, FP, TN, verbose=False):
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
    
    if verbose:
        print(f"  H(X)    = {H_X:.4f} bits")
        print(f"  H(X|Y)  = {H_XY:.4f} bits")
        print(f"  I(X;Y)  = {MI:.4f} bits")
    
    return MI, H_X, H_XY


def compute_metrics(TP, FN, FP, TN):
    """Compute all metrics from confusion matrix"""
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    precision_fake = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_fake = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0
    
    precision_legit = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_legit = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_legit = 2 * precision_legit * recall_legit / (precision_legit + recall_legit) if (precision_legit + recall_legit) > 0 else 0
    
    macro_f1 = (f1_fake + f1_legit) / 2
    
    return {
        'accuracy': accuracy,
        'precision_fake': precision_fake,
        'recall_fake': recall_fake,
        'f1_fake': f1_fake,
        'precision_legit': precision_legit,
        'recall_legit': recall_legit,
        'f1_legit': f1_legit,
        'macro_f1': macro_f1
    }


def task1_mutual_information():
    """TASK 1: Compute MI using professor's formula"""
    print("\n" + "="*60)
    print("TASK 1: MUTUAL INFORMATION")
    print("Formula: I(X;Y) = H(X) - H(X|Y) using log2 (bits)")
    print("="*60)
    
    mi_values = {}
    
    for setting in ['RAW', 'SMOTE']:
        print(f"\n{'='*60}")
        print(f"{setting} SETTING:")
        print(f"{'='*60}")
        mi_values[setting] = {}
        
        for model in ['RandomForest', 'SVM', 'MLP']:
            TP, FN, FP, TN = confusion_matrices[setting][model]
            print(f"\n{model}:")
            print(f"  TP={TP}, FN={FN}, FP={FP}, TN={TN}")
            mi, H_X, H_XY = compute_mutual_information(TP, FN, FP, TN, verbose=True)
            mi_values[setting][model] = mi
    
    return mi_values


def task2_normalized_weights(mi_values):
    """TASK 2: Compute normalized MI weights"""
    print("\n" + "="*60)
    print("TASK 2: NORMALIZED MI WEIGHTS")
    print("="*60)
    
    weights = {}
    
    for setting in ['RAW', 'SMOTE']:
        print(f"\nSETTING: {setting}")
        total_mi = sum(mi_values[setting].values())
        weights[setting] = {}
        
        for model in ['RandomForest', 'SVM', 'MLP']:
            weight = mi_values[setting][model] / total_mi
            weights[setting][model] = weight
            print(f"  {model:15s}: MI={mi_values[setting][model]:.6f}  Weight={weight:.4f}")
    
    return weights


def task3_full_metrics_table():
    """TASK 3: Compute and print full metrics table"""
    print("\n" + "="*60)
    print("TASK 3: FULL METRICS TABLE")
    print("="*60)
    
    all_metrics = {}
    
    for setting in ['RAW', 'SMOTE']:
        print(f"\n{'='*60}")
        print(f"SETTING: {setting}")
        print(f"{'='*60}")
        
        all_metrics[setting] = {}
        
        for model in ['RandomForest', 'SVM', 'MLP']:
            TP, FN, FP, TN = confusion_matrices[setting][model]
            metrics = compute_metrics(TP, FN, FP, TN)
            all_metrics[setting][model] = metrics
            
            setting_label = setting.lower()
            print(f"{setting_label} | {model:15s} | Acc={metrics['accuracy']:.4f} | "
                  f"Fake: P={metrics['precision_fake']:.4f} R={metrics['recall_fake']:.4f} "
                  f"F1={metrics['f1_fake']:.4f} | "
                  f"Legit: P={metrics['precision_legit']:.4f} R={metrics['recall_legit']:.4f} "
                  f"F1={metrics['f1_legit']:.4f} | "
                  f"MacroF1={metrics['macro_f1']:.4f}")
    
    return all_metrics


def task4_result_analysis(all_metrics, mi_values):
    """TASK 4: Automated result analysis"""
    print("\n" + "="*60)
    print("TASK 4: RESULT ANALYSIS")
    print("="*60)
    
    # Find best model overall
    best_model = None
    best_macro_f1 = 0
    for setting in ['RAW', 'SMOTE']:
        for model in ['RandomForest', 'SVM', 'MLP']:
            macro_f1 = all_metrics[setting][model]['macro_f1']
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_model = (model, setting)
    
    print(f"\n1. BEST MODEL OVERALL:")
    print(f"   {best_model[0]} ({best_model[1]}) with MacroF1 = {best_macro_f1:.4f}")
    
    # Most improved by SMOTE
    improvements = {}
    for model in ['RandomForest', 'SVM', 'MLP']:
        raw_f1 = all_metrics['RAW'][model]['macro_f1']
        smote_f1 = all_metrics['SMOTE'][model]['macro_f1']
        improvements[model] = smote_f1 - raw_f1
    
    most_improved = max(improvements, key=improvements.get)
    print(f"\n2. MOST IMPROVED BY SMOTE:")
    print(f"   {most_improved} with gain of {improvements[most_improved]:+.4f}")
    
    # SMOTE effect on each model
    print(f"\n3. SMOTE EFFECT ON EACH MODEL:")
    for model in ['RandomForest', 'SVM', 'MLP']:
        gain = improvements[model]
        effect = "helped" if gain > 0 else "hurt"
        print(f"   {model:15s}: {effect} by {abs(gain):.4f}")
    
    # Best for catching fake tasks
    best_fake_recall = 0
    best_fake_model = None
    for setting in ['RAW', 'SMOTE']:
        for model in ['RandomForest', 'SVM', 'MLP']:
            recall = all_metrics[setting][model]['recall_fake']
            if recall > best_fake_recall:
                best_fake_recall = recall
                best_fake_model = (model, setting)
    
    print(f"\n4. BEST FOR CATCHING FAKE TASKS (Highest Fake Recall):")
    print(f"   {best_fake_model[0]} ({best_fake_model[1]}) with Recall = {best_fake_recall:.4f}")
    
    # Most precise model
    best_fake_precision = 0
    best_precise_model = None
    for setting in ['RAW', 'SMOTE']:
        for model in ['RandomForest', 'SVM', 'MLP']:
            precision = all_metrics[setting][model]['precision_fake']
            if precision > best_fake_precision:
                best_fake_precision = precision
                best_precise_model = (model, setting)
    
    print(f"\n5. MOST PRECISE MODEL (Highest Fake Precision):")
    print(f"   {best_precise_model[0]} ({best_precise_model[1]}) with Precision = {best_fake_precision:.4f}")
    
    # MI ranking
    print(f"\n6. MI RANKING:")
    for setting in ['RAW', 'SMOTE']:
        print(f"   {setting}:")
        sorted_mi = sorted(mi_values[setting].items(), key=lambda x: x[1], reverse=True)
        for rank, (model, mi) in enumerate(sorted_mi, 1):
            print(f"      {rank}. {model:15s}: MI = {mi:.6f} bits")


def task5_comparison_chart(all_metrics, mi_values):
    """TASK 5: Create combined comparison bar chart"""
    print("\n" + "="*60)
    print("TASK 5: GENERATING COMPARISON CHART")
    print("="*60)
    
    models = ['RandomForest', 'SVM', 'MLP']
    x_labels = ['RF_RAW', 'RF_SMOTE', 'SVM_RAW', 'SVM_SMOTE', 'MLP_RAW', 'MLP_SMOTE']
    
    # Prepare data
    accuracy_data = []
    fake_f1_data = []
    macro_f1_data = []
    mi_data = []
    colors = []
    
    for model in models:
        for setting in ['RAW', 'SMOTE']:
            accuracy_data.append(all_metrics[setting][model]['accuracy'])
            fake_f1_data.append(all_metrics[setting][model]['f1_fake'])
            macro_f1_data.append(all_metrics[setting][model]['macro_f1'])
            mi_data.append(mi_values[setting][model])
            colors.append('steelblue' if setting == 'RAW' else 'orange')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Accuracy
    bars1 = axes[0, 0].bar(range(6), accuracy_data, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Model', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 0].set_title('Accuracy Comparison', fontweight='bold', fontsize=14)
    axes[0, 0].set_xticks(range(6))
    axes[0, 0].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, accuracy_data)):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Subplot 2: Fake F1
    bars2 = axes[0, 1].bar(range(6), fake_f1_data, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Model', fontweight='bold')
    axes[0, 1].set_ylabel('Fake F1 Score', fontweight='bold')
    axes[0, 1].set_title('Fake F1 Comparison', fontweight='bold', fontsize=14)
    axes[0, 1].set_xticks(range(6))
    axes[0, 1].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 1.1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, fake_f1_data)):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Subplot 3: MacroF1
    bars3 = axes[1, 0].bar(range(6), macro_f1_data, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Model', fontweight='bold')
    axes[1, 0].set_ylabel('Macro F1 Score', fontweight='bold')
    axes[1, 0].set_title('MacroF1 Comparison', fontweight='bold', fontsize=14)
    axes[1, 0].set_xticks(range(6))
    axes[1, 0].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars3, macro_f1_data)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Subplot 4: MI Values
    bars4 = axes[1, 1].bar(range(6), mi_data, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Model', fontweight='bold')
    axes[1, 1].set_ylabel('Mutual Information (bits)', fontweight='bold')
    axes[1, 1].set_title('MI Values Comparison', fontweight='bold', fontsize=14)
    axes[1, 1].set_xticks(range(6))
    axes[1, 1].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars4, mi_data)):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='RAW'),
                      Patch(facecolor='orange', alpha=0.7, label='SMOTE')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12, 
              bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('Data/phase3_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: Data/phase3_comparison.png")
    plt.close()


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("PHASE 3: RESULTS SUMMARY AND ANALYSIS")
    print("Using Professor's MI Formula: I(X;Y) = H(X) - H(X|Y)")
    print("="*60)
    
    # Task 1: Mutual Information
    mi_values = task1_mutual_information()
    
    # Task 2: Normalized Weights
    weights = task2_normalized_weights(mi_values)
    
    # Task 3: Full Metrics Table
    all_metrics = task3_full_metrics_table()
    
    # Task 4: Result Analysis
    task4_result_analysis(all_metrics, mi_values)
    
    # Task 5: Comparison Chart
    task5_comparison_chart(all_metrics, mi_values)
    
    print("\n" + "="*60)
    print("PHASE 3 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated Files:")
    print("  • Data/phase3_comparison.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
