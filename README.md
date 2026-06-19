# 🌱 Dry Bean Classification using Machine Learning

## 📌 Project Overview

This project focuses on building a multiclass classification model to identify the type of dry bean based on its physical and geometric characteristics.

The project covers the complete machine learning workflow, including:

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Scaling
* Model Training and Evaluation
* Model Comparison
* Model Selection
* Streamlit Deployment

---

## 📊 Dataset Description

The Dry Bean Dataset contains numerical features describing the shape and size characteristics of beans.

### Features

* Area
* Perimeter
* MajorAxisLength
* MinorAxisLength
* AspectRatio
* Eccentricity
* ConvexArea
* EquivDiameter
* Extent
* Solidity
* Roundness
* Compactness
* ShapeFactor1
* ShapeFactor2
* ShapeFactor3
* ShapeFactor4

### Target Classes

The target variable consists of seven bean varieties:

* SEKER
* BARBUNYA
* BOMBAY
* CALI
* DERMASON
* HOROZ
* SIRA

Since the target variable is categorical, Label Encoding was applied before model training.

---

## ⚙️ Data Preprocessing

### Missing Value Analysis

* Checked the dataset for missing values.
* No significant missing values were found.

### Feature Scaling

Several machine learning algorithms are sensitive to feature magnitude. Therefore, feature scaling was performed using:

```python
StandardScaler()
```

Feature scaling significantly improved the performance of:

* Support Vector Classifier (SVC)
* K-Nearest Neighbors (KNN)

### Target Encoding

The target labels were converted into numerical values using:

```python
LabelEncoder()
```

---

## 🤖 Models Evaluated

The following machine learning algorithms were trained and compared:

| Model                           | Description                     |
| ------------------------------- | ------------------------------- |
| Logistic Regression             | Baseline multiclass classifier  |
| Decision Tree Classifier        | Tree-based classification model |
| Random Forest Classifier        | Ensemble of decision trees      |
| Bagging Classifier              | Bootstrap aggregation ensemble  |
| K-Nearest Neighbors (KNN)       | Distance-based classifier       |
| XGBoost Classifier              | Gradient boosting algorithm     |
| Support Vector Classifier (SVC) | Margin-based classifier         |

---

## 📈 Model Comparison

Multiple classification algorithms were evaluated using the same train-test split and preprocessing pipeline.

The models compared were:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Bagging Classifier
* K-Nearest Neighbors (KNN)
* XGBoost Classifier
* Support Vector Classifier (SVC)

After experimentation, **Support Vector Classifier (SVC)** achieved the best overall performance when combined with **StandardScaler**.

---

## 🏆 Best Performing Model

### Support Vector Classifier (SVC)

Final pipeline used:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(
        kernel='rbf',
        probability=True,
        gamma = 'scale',
        C = 10,
        random_state=42
    ))
])
```

### Why SVC Performed Best

* Handles complex and non-linear decision boundaries effectively.
* Performs well in high-dimensional feature spaces.
* Robust against overfitting when properly configured.
* Benefited significantly from feature standardization.
* Produced better class separation than the other evaluated models.

---

## 📋 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

### Final Results

| Metric          | Value                           |
| --------------- | ------------------------------- |
| Best Model      | Support Vector Classifier (SVC) |
| Feature Scaling | StandardScaler                  |
| ROC-AUC Score   | **93.58%**                      |

The ROC-AUC score of **93.25%** demonstrates strong discriminative performance across all bean classes and confirms the effectiveness of the selected model.

---

## 🚀 Streamlit Deployment

The final model was deployed using **Streamlit**, allowing users to interactively predict bean classes through a web interface.

### Features

* User-friendly interface
* Manual feature input
* Real-time predictions
* Prediction probability display
* Bean class identification

---

## 📁 Project Structure

```text
Beans-Classification/
│
├── Dry_Bean_Dataset.csv
├── model_training.ipynb
├── svc_pipeline.pkl
├── label_encoder.pkl
├── app.py
├── requirements.txt
└── README.md
```

### File Description

| File                 | Description                                  |
| -------------------- | -------------------------------------------- |
| Dry_Bean_Dataset.csv | Dataset used for training and evaluation     |
| model_training.ipynb | EDA, preprocessing, training, and evaluation |
| svc_pipeline.pkl     | Trained SVC pipeline with StandardScaler     |
| label_encoder.pkl    | Saved LabelEncoder for decoding predictions  |
| app.py               | Streamlit application                        |
| requirements.txt     | Project dependencies                         |
| README.md            | Project documentation                        |

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## 🎯 Conclusion

This project demonstrates a complete machine learning workflow for multiclass classification using the Dry Bean Dataset.

Multiple machine learning algorithms were trained, evaluated, and compared to identify the most effective model.

Among all tested algorithms, **Support Vector Classifier (SVC)** combined with **StandardScaler** delivered the strongest performance, achieving a **ROC-AUC Score of 93.58%**.

The final model was successfully deployed using **Streamlit**, providing an interactive web application for predicting dry bean varieties based on their physical characteristics.
