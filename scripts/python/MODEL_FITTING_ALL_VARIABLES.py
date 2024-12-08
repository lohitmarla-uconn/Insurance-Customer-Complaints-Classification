"""
Title: Optimizing Insurance Complaint Classification - All Variables
Format: Python Script
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the data
file_path = "data/Insurance_complaints__All_data.csv"
insurance_df = pd.read_csv(file_path)

# Define all categorical columns and their handling
categorical_columns = insurance_df.select_dtypes(include=["object", "category"]).columns
numerical_columns = insurance_df.select_dtypes(include=["number"]).columns

# Handling missing values
categorical_imputer = SimpleImputer(strategy="most_frequent")
numerical_imputer = SimpleImputer(strategy="mean")

# Encoding categorical variables
encoder = LabelEncoder()
for col in categorical_columns:
    insurance_df[col] = encoder.fit_transform(insurance_df[col].astype(str))

# Splitting data into features and target
target_column = "target"  # Replace with the actual target column name
X = insurance_df.drop(target_column, axis=1)
y = insurance_df[target_column]

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Naive Bayes Model
print("\nTraining Naive Bayes Model...")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predictions for Naive Bayes
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

# Logistic Regression Model
print("\nTraining Logistic Regression Model...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Predictions for Logistic Regression
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Decision Tree Model
print("\nTraining Decision Tree Model...")
dt_model = DecisionTreeClassifier(random_state=123)
dt_model.fit(X_train, y_train)

# Predictions for Decision Tree
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# Random Forest Model
print("\nTraining Random Forest Model...")
rf_model = RandomForestClassifier(random_state=123)
rf_model.fit(X_train, y_train)

# Predictions for Random Forest
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot ROC Curves for All Models
plt.figure(figsize=(10, 8))
models = {
    "Naive Bayes": nb_model,
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
}
for model_name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend()
plt.show()
