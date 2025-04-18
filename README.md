# EECS486 Final Project

## Overview
In this project, we propose a machine‑learning framework to predict social media addiction levels (low, moderate, high)  using behavioral and self‑reported features. The implementation includes baseline and improved pipelines, advanced ensembling, and visualizations.

## Paper Reference
This is the code submission for our paper.

## Features
- **Baseline Random Forest** with stratified 80/20 split, classification report, macro‑AUROC, and diagnostic plots.
- **Hyperparameter‑tuned Gradient Boosting** (HistGradientBoostingClassifier) with SMOTENC oversampling under `RandomizedSearchCV` 
- **Cost‑Sensitive Stacking Ensemble** combining Random Forest and Logistic Regression for high‑risk user detection
- **Visualization utilities** for distribution charts, correlation heatmaps, polar plots, and pipeline flowchart

## Installation

### Prerequisites
- Python 3.13.2  
- [Graphviz](https://graphviz.org/) (for generating flowcharts)

### Python packages
```bash
pip install pandas numpy scikit-learn imbalanced-learn category_encoders matplotlib graphviz
```

## Repository Structure
.
├── baseline_model.py
├── models_improved.py
├── visualize.py
├── data/
│   └── social_media_dataset.csv
├── plots/
├── results/
└── 486_Final_Report.pdf

## Usage
1. **Run baseline model**: 
    - Prints classification report & macro‑AUROC
    - Saves productivity‑loss bar chart, confusion matrix, ROC curve under plots/
    ```bash
     python baseline_model.py
    ```
2. **Run improved model**
    - Prints CV logs
    - Prints GBDT & stacking performance
    - Saves confusion matrices under plots/
    ```bash
    python models_improved.py
    ```    
3. **generate visualization of datasets**
    - Plots `ProdutivityLoss` Distribution, 
    - Plots correlation heatmap, 
    - Plots polar chart for `AdditionLevel`
    - Plots pipeline flowchart
    ```bash
    python visualize.py
    ```
## Acknowledgments
We gratefully acknowledge the authors and maintainers of the [Time Wasters on Social Media dataset](https://www.kaggle.com/datasets/muhammadroshaanriaz/time-wasters-on-social-media?resource=download) for sharing their synthetic user survey on Kaggle.