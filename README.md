# embc2024.py

## Overview

`embc2024.py` is a Python script that is part of a larger project. This script is responsible for performing various data processing, feature extraction, and machine learning tasks. It leverages several Python libraries to achieve this, including numpy, pandas, tsfel, tsfresh, pycatch22, imblearn, and sklearn. The data used for this analysis can be found in [Zenodo]()

## Features

- **Data Processing**: The script uses numpy and pandas for handling and manipulating data. These libraries provide robust data structures and functions for efficient data processing.

- **Feature Extraction**: The script uses tsfel, tsfresh, and pycatch22 for feature extraction from time-series data. These libraries provide comprehensive tools for extracting useful features from time-series data.

- **Data Scaling**: The script uses StandardScaler, RobustScaler, and MinMaxScaler from sklearn.preprocessing for scaling features. These methods standardize the features by removing the mean and scaling to unit variance, scale features using statistics that are robust to outliers, and transform features by scaling each feature to a given range, respectively.

- **Dimensionality Reduction**: The script uses PCA (Principal Component Analysis) from sklearn.decomposition for reducing the dimensionality of the dataset. This technique transforms the data to a new coordinate system such that the greatest variance by some projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.

- **Oversampling**: The script uses SMOTE (Synthetic Minority Over-sampling Technique) from imblearn.over_sampling for balancing the dataset. This technique generates synthetic samples from the minority class.

- **Feature Selection**: The script uses various feature selection techniques from sklearn.feature_selection, including VarianceThreshold, SelectKBest, SelectPercentile, f_classif, mutual_info_classif, and SelectFromModel. These techniques are used to select the most relevant features for the model.

- **Model Evaluation**: The script uses various metrics from sklearn.metrics for evaluating the performance of the model, including accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, matthews_corrcoef, and roc_curve.

## Usage

To run the script, use the following command:

```bash
python embc2024.py
```

Additional options for the script is documented in code.

## Dependencies

This script requires the following Python packages:

- numpy
- pandas
- tsfel
- tsfresh
- pycatch22
- imblearn
- sklearn

You can install these using pip:

```bash
pip install numpy pandas tsfel tsfresh pycatch22 imbalanced-learn scikit-learn
```
