# EECS486 Final Project

## Overview
In this project, we propose a machine learning framework to predict social media addiction levels (low, moderate, high) using behavioral and self-reported features. The implementation includes baseline and improved pipelines, advanced ensembling strategies, and visualization utilities.

## Paper Reference
This repository contains the code submitted alongside our final report for EECS 486.

## Features
- **Baseline Random Forest** with stratified 80/20 split, classification report, macro‑AUROC, and diagnostic plots
- **Hyperparameter‑tuned Gradient Boosting** (`HistGradientBoostingClassifier`) with `SMOTENC` oversampling via `RandomizedSearchCV`
- **Cost‑Sensitive Stacking Ensemble** combining Random Forest and Logistic Regression for high‑risk user detection
- **Visualization tools** including distribution plots, correlation heatmaps, polar charts, and a model pipeline diagram

## Installation

### Prerequisites
- Python 3.6 or higher [Recommended: Python 3.13.2]
- [Graphviz](https://graphviz.org/) installed system-wide for rendering pipeline flowcharts:
  - **Ubuntu**: `sudo apt-get install graphviz`
  - **macOS**: `brew install graphviz`

### Install Python Dependencies
To install all required Python packages, run:

```bash
pip install --user -r requirements.txt
```

Or, using a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Repository Structure

```
.
├── baseline_model.py
├── models_improved.py
├── visualize.py
├── requirements.txt
├── data/
│   └── social_media_dataset.csv
├── plots/
├── results/
└── 486_Final_Report.pdf
```

## Usage

### 1. Run Baseline Model
Prints classification report and macro‑AUROC.  
Saves productivity-loss bar chart, confusion matrix, and ROC curve to `plots/`.

```bash
python baseline_model.py
```

### 2. Run Improved Model
Prints CV logs and stacking performance.  
Saves confusion matrices to `plots/`.

```bash
python models_improved.py
```

### 3. Run Visualizations
- Plots distribution of `ProductivityLoss`
- Plots correlation heatmap
- Plots polar chart for `AddictionLevel`
- Generates pipeline flowchart using `graphviz`

```bash
python visualize.py
```

## Acknowledgments
We gratefully acknowledge the authors and maintainers of the [Time Wasters on Social Media dataset](https://www.kaggle.com/datasets/muhammadroshaanriaz/time-wasters-on-social-media?resource=download) for sharing their synthetic user survey on Kaggle.
