# Churn-Prediction.py

# Retail Customer Churn Prediction

This project explores customer churn prediction using real-world e-commerce transaction data. The goal is to identify customers at risk of leaving by analysing historical purchase patterns and behaviour.

## Project Structure

# Project Overview

Customer churn is a key business challenge in the retail and e-commerce space. By analysing past customer behaviour, we aim to:
- Define churn based on inactivity or reduced spend
- Use ML models to predict which customers are at risk
- Explore key factors like recency, frequency, and spend habits


## Approach and Methodology
  
1. **Data Loading:** Read raw transaction data from CSV.
2. **Data Cleaning:** Remove duplicate records, handle missing values, and filter invalid entries.
3. **Feature Engineering:** Generate features such as recency, frequency, monetary spend, and average spend based on purchase history.
4. **Model Building:** Use a Random Forest classifier with hyperparameter tuning and cross-validation.
5. **Evaluation:** Measure model accuracy, generate confusion matrix, and visualise key features.




## ⚙️ Environment Setup

We use Conda to manage dependencies in an isolated environment.

### Step 1: Create a Conda Environment

```
conda create --name dspt python=3.10
```

### Step 2: Activate The Enviroment 
### Conda is recommended


```
conda activate dspt
```

### Step 2: Install dependencies

After creating and activating your environment, install the necessary packages. You can install packages manually, or use the `requirements.txt` file 

Then run:

```bash
pip freeze > requirements.txt
```

**Dataset Download:**
Please download the dataset from https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data. Save it as `online_retail_II.csv` in your project folder before running the scripts.


