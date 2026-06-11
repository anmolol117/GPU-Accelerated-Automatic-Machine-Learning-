
# GPU-Accelerated AutoML Pipeline

An intelligent **GPU-Accelerated Automated Machine Learning (AutoML) Pipeline** designed to automate the complete machine learning workflow — from data preprocessing to model selection, hyperparameter tuning, evaluation, and model saving with minimal human intervention.

The system supports both **Classification** and **Regression** tasks and leverages **GPU acceleration using cuML and XGBoost** to improve computational efficiency and scalability.

---

## Overview

Machine learning workflows often require multiple manual steps such as:

- Data preprocessing  
- Feature engineering  
- Feature selection  
- Model selection  
- Hyperparameter tuning  
- Model evaluation  

This project automates the **end-to-end ML pipeline**, reducing manual effort, improving reproducibility, and maximizing performance.

---

## Pipeline Architecture

The following flowchart illustrates the complete AutoML workflow:

![GPU Accelerated AutoML Pipeline](Gemini_Generated_Image_e2jo12e2jo12e2jo.png)

---

## ✨ Features

### Automatic Data Preprocessing
- Missing value handling using **KNN Imputation**
- Categorical encoding using **Label Encoding**
- GPU-compatible data conversion using **cuDF**
- Intelligent preprocessing based on dataset properties

### Feature Selection
A two-stage feature selection strategy:

1. **Variance Thresholding**
   - Removes low-variance features (`< 0.01`)

2. **Correlation-Based Selection**
   - Selects top correlated features with respect to the target variable

### Evolutionary Feature Engineering
Automatically generates new features through:

- Feature multiplication (`f1 × f2`)
- Feature addition (`f1 + f2`)
- Tree-based feature importance ranking
- Iterative refinement cycles

### Dynamic Model Selection
Model selection is performed dynamically based on:

- Problem type (**Classification / Regression**)
- Dataset size
- Hardware availability (**GPU vs CPU**)

### Hyperparameter Optimization
- Randomized parameter search
- Multiple optimization trials
- Automatic best parameter selection

### Automatic Evaluation & Best Model Selection
**Classification Tasks**
- Accuracy-based evaluation

**Regression Tasks**
- Mean Squared Error (MSE)-based evaluation

### Model Persistence
- Best model saved using **Pickle/Joblib**
- Ready for deployment and reuse

---

## Models Supported

The pipeline supports multiple ML algorithms:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **XGBoost**
- **LightGBM**

---

## Tech Stack

### Libraries & Frameworks
- **Python**
- **cuDF**
- **cuML**
- **XGBoost**
- **LightGBM**
- **Scikit-learn**
- **NumPy**
- **Pandas**

### Hardware Support
- **GPU Accelerated Training**
- **CPU Fallback Support**

---

## Dataset Characteristics

The system was tested on multiple CSV datasets to ensure robustness.

| Feature | Details |
|----------|----------|
| Number of datasets | 6 |
| Rows per dataset | 10,000+ |
| Number of features | 6–18 |
| Data types | Numerical & Categorical |
| Tasks supported | Classification & Regression |

### Common Dataset Properties
- Mixed numerical and categorical features
- Missing values and noisy data
- Class imbalance in classification datasets
- Non-uniform feature distributions

---

## Results

The AutoML system demonstrated strong and consistent performance across multiple datasets.

### Classification Performance
**Accuracy:** `85% – 97%`

### Regression Performance
**Accuracy:** `85% – 91%`

### Key Observations
- **Tree-based models** such as Random Forest, XGBoost, and LightGBM performed consistently well on larger datasets.
- **KNN and SVM** achieved good performance on smaller and structured datasets.
- Feature engineering and selection improved overall model performance.

---

## Future Scope

Potential improvements include:

- Support for additional formats (**Excel, JSON, SQL/NoSQL databases**)
- Deep learning model integration
- Time-series forecasting support
- Image-based AutoML systems
- REST API deployment
- Web-based user interfaces for production usage

---

## Team Members

- **Anmol Mangat** — 102303369  
- **Mehar Walia** — 102303223  
- **Bhoomika** — 102303571  
- **Sarvpreet Kaur** — 102317071  

**Supervisor:** Dr. Manisha Malik

---

## License

This project is developed for academic and research purposes.
