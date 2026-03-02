# Ubiquitous Sensing - Assignment 2

Binary Classification for Fake Task Detection using Machine Learning

## Project Structure

```
scripts/
├── Data/                          # Data folder (not tracked in git)
│   ├── train.csv
│   ├── test.csv
│   ├── train_preprocessed.csv
│   └── test_preprocessed.csv
├── phase1_data_analysis.py        # Data loading and preprocessing
└── utils.py                       # Utility functions
├── model_random_forest.py         # Random Forest classifier
├── model_svm.py                   # SVM classifier
├── model_mlp_pytorch.py           # PyTorch MLP (100 epochs, RAW & SMOTE)
├── phase3_results_summary.py      # Results visualization

```

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn torch
```

## Usage

### 1. Data Preprocessing
```bash
python3 phase1_data_analysis.py
```

### 2. Train Individual Models

**Random Forest:**
```bash
python3 model_random_forest.py
```

**SVM:**
```bash
python3 model_svm.py
```

**PyTorch MLP (100 epochs with RAW & SMOTE):**
```bash
python3 model_mlp_pytorch.py
```


## Models

- **Random Forest**: 100 trees, trained on RAW and SMOTE data
- **SVM**: RBF kernel, trained on RAW and SMOTE data
- **PyTorch MLP**: (100, 50) hidden layers, 100 epochs, trained on RAW and SMOTE data

## Results

All models generate:
- Confusion matrices (saved as PNG)
- Detailed metrics: Accuracy, Precision, Recall, F1-Score (per class, macro, weighted)
- Training history plots (PyTorch MLP only)

## Notes

- Data folder is excluded from git tracking
- SMOTE (BorderlineSMOTE) is applied for handling class imbalance
- PyTorch models are saved as `.pth` files (also excluded from git)
# ML-legitimate-task-classifier
