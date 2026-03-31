# ☄️ Asteroid Hazard Prediction

A Machine Learning project to predict whether a Near-Earth asteroid 
is Potentially Hazardous using NASA's open asteroid dataset.

## 📌 Problem Statement
NASA tracks thousands of Near-Earth Objects (asteroids). This project 
builds an ML classifier to automatically identify Potentially Hazardous 
Asteroids (PHAs) based on their orbital and physical characteristics.

## 📊 Dataset
- Source: NASA NeoWs (Near Earth Object Web Service)
- Records: 4,687 asteroids
- Features: 15 orbital & physical parameters
- Target: `Hazardous` (Binary Classification)

## 🔧 Tech Stack
- Python, Pandas, Scikit-learn, XGBoost
- SMOTE (imbalanced-learn)
- SHAP (Explainability)
- Matplotlib, Seaborn

## 🚀 Approach
1. EDA & Data Cleaning
2. Feature Engineering & Selection
3. Class Imbalance Handling (SMOTE)
4. Model Training & Comparison
5. SHAP Explainability

## 📈 Results
| Model | Accuracy | F1 Score (Hazardous) |
|---|---|---|
| Random Forest | 99.8% | 0.99 |
| XGBoost | 99.6% | 0.99 |
| Neural Network | 83.0% | 0.44 |

✅ Best Model: Random Forest

## 🔍 Key Insight (SHAP)
Minimum Orbit Intersection Distance is the most critical predictor — 
aligning with NASA's own definition of a Potentially Hazardous Asteroid.
