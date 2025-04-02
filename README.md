# _Hi, I'm Saurabh! 👋 This is my Capston Project 🎯!_

## 🦷 Forensic Dentistry: Using Dental Metrics to Predict Gender



## 📖 Project Overview
Forensic dentistry is a branch of forensic medicine that helps identify individuals using dental measurements. This project utilizes **machine learning** techniques to predict **gender** based on dental metrics.

## 🚀 Tech Stack
- **Programming Language**: Python
- **Tools**: Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
- **Project Difficulty level** : Rookie/ Basic
## 🎯 Objectives
- **Analyze** dental data and its relationship with gender.
- **Implement machine learning models** for gender classification.
- **Evaluate and compare model performance**.

## 📂 Dataset Description
**File:** `Dentistry Dataset.csv`

| Feature Name                           | Description |
|----------------------------------------|-------------|
| **Age**                                | The age of the individual |
| **Gender (Target Variable)**           | Male (1) / Female (0) |
| **Inter-canine distance intraoral**    | Measurement between upper canine teeth |
| **Right & Left Canine Width Casts**    | Width of the right and left canines |
| **Canine Index**                       | Canine index measurement |

## 🛠 Methodology
### 1️⃣ **Data Preprocessing**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer

# Load dataset
df = pd.read_csv("Dentistry Dataset.csv")

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Encode categorical data
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Normalize features
normalizer = Normalizer()
df.iloc[:, 2:] = normalizer.fit_transform(df.iloc[:, 2:])
```

### 2️⃣ **Exploratory Data Analysis (EDA)**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap for correlation analysis
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
```

### 3️⃣ **Model Building**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Splitting dataset
X = df.drop(columns=['Gender'])
y = df['Gender']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## 📊 Results & Analysis
| Model                  | Accuracy |
|------------------------|-------------|
| Logistic Regression    | 0.64090909090909         |
| Decision Tree         | 0.877272727272727         |
| Random Forest        | 0.895454545454545         |
| XGBoost             |     0.9         |

📌 **Best Performing Model:** _XGBoost (XX% Accuracy)_

## 📢 Conclusion & Future Work
- **XGBoost performed the best**, showing potential for forensic applications.
- **Future work:** More diverse datasets, deep learning techniques, real-world testing.

## 📂 Installation & Usage
### 🔹 **Clone the Repository**
```bash
git clone https://github.com/Saurabhji-1/Capstone-Project/tree/main
```

### 🔹 **Install Dependencies**
```bash
!pip install numpy pandas matplotlib seaborn scikit-learn xgboost

```

### 🔹 **Run the Jupyter Notebook**
```bash
jupyter notebook
```

## 📂 References
- **Scikit-learn Documentation**: [https://scikit-learn.org](https://scikit-learn.org)
- **XGBoost Documentation**: [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)

---

📌 **Author:** _**Saurabh Sharma**_  

📌 **GitHub Repository:** _https://github.com/Saurabhji-1?tab=repositories_

