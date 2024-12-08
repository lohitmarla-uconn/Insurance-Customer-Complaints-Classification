# Insurance Customer Complaints Classification

## Overview

This project focuses on analyzing and predicting the nature of customer complaints within the insurance domain. By leveraging data cleaning, exploratory analysis, and machine learning models, the project aims to classify complaint types and provide actionable insights for improving customer service and operational efficiency.

The project implements models like **Random Forest, Logistic Regression, Naive Bayes**, and **Decision Trees** to predict complaint types and assess factors contributing to customer dissatisfaction.

---

## Table of Contents
1. [Objectives](#objectives)
2. [Dataset Description](#dataset-description)
3. [Technologies Used](#technologies-used)
4. [Methodology](#methodology)
5. [Project Structure](#project-structure)
6. [How to Run](#how-to-run)
7. [Key Findings](#key-findings)
8. [Future Enhancements](#future-enhancements)

---

## Objectives

- **Enhance Customer Service**: Identify patterns in complaint resolution to improve response efficiency.
- **Optimize Resource Allocation**: Understand complaint distribution across types to allocate resources effectively.
- **Risk Management**: Detect operational vulnerabilities linked to complaint trends.
- **Product Development**: Use insights to guide the improvement of insurance products and services.

---

## Dataset Description

The dataset contains detailed information about customer complaints, including:
- **Complaint Type** (target variable)
- **Filed Against**: Entity being complained about.
- **Reason for Complaint**: Issues like denial of claim or delays.
- **Coverage Type**: Type of insurance coverage involved.
- **How Resolved**: Method of resolution (e.g., information furnished, claim paid).
- **Dates**: Complaint receipt and closure dates.
- **Complaint Duration**: Time taken to resolve a complaint.

Data cleaning steps included:
- Standardizing column names.
- Handling missing values through imputation.
- Splitting multi-value columns into separate rows for granularity.

---

## Technologies Used

### Programming Languages
- Python
- R

### Libraries
- **Python**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **R**: discrim, tidymodels, nnet, dplyr, tidyr, naivebayes, caret, ggplot2

---

## Methodology

### 1. Data Cleaning and Preprocessing
- Standardized column names.
- Handled missing values through mode/mean imputation.
- Filtered specific complaint types for focused analysis.

### 2. Exploratory Data Analysis
- Visualized complaint trends and distribution by type.
- Generated word clouds to identify frequent keywords in complaints.
- Examined date differences to assess resolution efficiency.

### 3. Modeling
- Implemented classification models:
  - Random Forest
  - Logistic Regression
  - Naive Bayes
  - Decision Trees
- Evaluated models using metrics such as ROC AUC and accuracy.

### 4. Validation
- Used cross-validation techniques (k-fold, Monte Carlo, Bootstrapping) to ensure model robustness.
- Compared performance metrics across models.

---

## Project Structure

```
Insurance_Customer_Complaints_Classification/
├── data/
│   ├── Insurance_complaints__All_data.csv
│   ├── Cleaned_data.csv
├── scripts/
│   ├── python/
│   │   ├── MODEL_FITTING_ALL_VARIABLES.py
│   │   ├── stat_data_cleaning.ipynb
│   │   ├── messy_rows.ipynb
│   ├── r/
│       ├── MODEL_FITTING_SELECTED_VARIABLES.qmd
│       └── Insurance_Complaints_EDA.Rmd
├── reports/
│   └── Project_Report_Stats_Final.pdf
├── README.md
└── requirements.txt
```

---

## How to Run

### Prerequisites
1. **Install Python 3.8+**:
   - Install Python from [python.org](https://www.python.org/).
2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install R and Required Packages**:
   - Install R from [CRAN](https://cran.r-project.org/).
   - Install R packages:
     ```R
     install.packages(c("discrim", "tidymodels", "nnet", "dplyr", "tidyr", "naivebayes", "caret", "ggplot2"))
     ```

---

### Steps to Run the Project

#### Python Scripts
1. **Run Data Cleaning**:
   ```bash
   python scripts/python/stat_data_cleaning.ipynb
   ```
   - Cleans and preprocesses the dataset.

2. **Run Machine Learning Models**:
   ```bash
   python scripts/python/MODEL_FITTING_ALL_VARIABLES.py
   ```

#### R Scripts
1. **Run EDA**:
   Open `Insurance_Complaints_EDA.Rmd` in RStudio and execute the code chunks.

2. **Render Quarto Report**:
   Ensure Quarto is installed ([Quarto Installation Guide](https://quarto.org/docs/get-started/)):
   ```bash
   quarto render scripts/r/MODEL_FITTING_SELECTED_VARIABLES.qmd
   ```

   - The output report (HTML) will be saved in the `html/` folder.

---

## Key Findings

1. **Top-Performing Models**:
   - Random Forest achieved the highest accuracy and robustness across validation techniques.
   - Logistic Regression and Naive Bayes models also performed well.

2. **Complaint Insights**:
   - Certain complaint types like "Teacher Retirement System" and "Workers Compensation Network" were predominant.
   - Keywords like "unauthorized" and "denied" highlighted recurring issues.

3. **Resolution Trends**:
   - Average resolution time varied significantly across complaint types, revealing opportunities for process improvement.

---

## Future Enhancements

1. **Integrate Text Analysis**:
   - Use NLP models like BERT for in-depth analysis of complaint text.
2. **Dashboard Development**:
   - Create an interactive dashboard for real-time monitoring of complaint trends.
3. **Broaden Scope**:
   - Include external datasets (e.g., customer feedback, financial records) to enhance predictive accuracy.

---