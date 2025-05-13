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

def create_features(df):
    """
    Create features for churn prediction using RFM model (Recency, Frequency, Monetary).
    """
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    print("InvoiceDate dtype after conversion:", df['InvoiceDate'].dtype)

    # Check and handle invalid dates
    invalid_dates = df['InvoiceDate'].isna().sum()
    if invalid_dates > 0:
        print(f"Warning: There are {invalid_dates} invalid dates in the 'InvoiceDate' column.")
        # Optionally, drop rows with invalid dates (if they are not critical)
        df = df.dropna(subset=['InvoiceDate'])

    cutoff_date = pd.to_datetime("2011-12-10")  # or another fixed date before last invoice date

# Ensure that InvoiceDate is datetime
    if not np.issubdtype(df['InvoiceDate'].dtype, np.datetime64):
      print("Error: InvoiceDate is not datetime!")
      return df  # Stop processing if InvoiceDate is not datetime

 # Confirm InvoiceDate is datetime
    if not np.issubdtype(df['InvoiceDate'].dtype, np.datetime64):
       print("Error: InvoiceDate is not in datetime format!")
       return df  # Optional safeguard


# Calculate Recency as days since last purchase relative to cutoff_date
    df['Recency'] = (cutoff_date - df['InvoiceDate']).dt.days


    # Group by Customer ID and aggregate features
    customer_data = df.groupby('Customer ID').agg({
        'Recency': 'min',  
        'Invoice': 'count', 
        'Price': 'sum',  
    }).reset_index()

    # Rename columns 
    customer_data.rename(columns={'Invoice': 'Frequency', 'Price': 'Monetary'}, inplace=True)

    # Average Spend per Transaction
    customer_data['Average_Spend'] = customer_data['Monetary'] / customer_data['Frequency']

    # Create a binary churn column: customers with no purchases in the last 180 days are considered churned
    customer_data['Churn'] = np.where(customer_data['Recency'] > 180, 1, 0)  # 180 days = 6 months

    print(f"Features Created: {customer_data.shape[1]} columns")
    return customer_data

# Build and Train the Model
def build_model(df):
    """
    Build and train a Random Forest model to predict customer churn.
    """
    # Split data into features (X) and target (y)
    X = df[['Recency', 'Frequency', 'Monetary', 'Average_Spend']]
    y = df['Churn']

    # Split the data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing Pipeline 
    numerical_features = ['Recency', 'Frequency', 'Monetary', 'Average_Spend']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Hyperparameter Tuning with GridSearchCV
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5, 10]
    }

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")

    y_pred = grid_search.best_estimator_.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
    print(f"Cross-validated Accuracy: {cv_scores.mean():.4f}")

    ConfusionMatrixDisplay.from_estimator(grid_search.best_estimator_, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.show() 


