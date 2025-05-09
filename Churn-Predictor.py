import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os 

def load_data():
    data_path = 'online_retail_II.csv'  
    df = pd.read_csv(data_path)
    print("✅ Data Loaded Successfully")
    print(df.head())
    return df

def clean_data(df):
    """
    Clean the retail dataset by handling missing values and removing duplicates.
    """
    print("Cleaning the Data...")

    df.drop_duplicates(inplace=True)

    # Handle missing values by filling them with 0
    df.fillna(0, inplace=True)

    # Remove negative values in 'Quantity' and 'Price'
    df = df[df['Quantity'] > 0]

    print(f"Cleaned DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")
    return df
