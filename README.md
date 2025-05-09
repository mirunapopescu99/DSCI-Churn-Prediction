# Churn-Prediction.py

# Retail Customer Churn Prediction

This project explores customer churn prediction using real-world e-commerce transaction data. The goal is to identify customers at risk of leaving by analysing historical purchase patterns and behaviour.

## Project Structure

# Project Overview

Customer churn is a key business challenge in the retail and e-commerce space. By analysing past customer behaviour, we aim to:
- Define churn based on inactivity or reduced spend
- Use ML models to predict which customers are at risk
- Explore key factors like recency, frequency, and spend habits

## ⚙️ Environment Setup

We use Conda to manage dependencies in an isolated environment.

### Step 1: Create a Conda Environment

```bash
conda create --name dspt python=3.10

### Step 2: Activate The Enviroment

```bash
conda activate dspt

### Step 3: Download Required Packes

```bash
conda install pandas numpy scikit-learn matplotlib seaborn jupyter

### Step 4: Add Jupitor Kernel 

```bash
python -m ipykernel install --user --name=dspt --display-name "Python (dspt)"


