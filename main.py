from typing import Optional
from data_loader import load_data
from data_cleaning import clean_data
from feature_engineering import create_features
from model import build_model
from config import CUTOFF_DATE, CHURN_WINDOW, RANDOM_STATE, TEST_SIZE

from sklearn.model_selection import train_test_split
import pandas as pd

def main() -> None:
    """
    Load data, clean, engineer features, split dataset, 
    and train the churn prediction model.
    """
    try:
        # Load raw data
        df: Optional[pd.DataFrame] = load_data()

        # Data cleaning
        cleaned_df = clean_data(df)
        print("Data cleaned.")

        # Create features and labels
        customer_data = create_features(
            cleaned_df, 
            cutoff_date_str=CUTOFF_DATE, 
            churn_window=CHURN_WINDOW
        )

        if customer_data is not None and not customer_data.empty:
            feature_cols = ['recency', 'frequency', 'monetary', 'average_spend']

            # Split into train/test
            train_df, test_df = train_test_split(
                customer_data,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=customer_data['churn']
            )
            print("Data split into train and test.")

            # Prepare feature matrices
            X_train = train_df[feature_cols]
            y_train = train_df['churn']
            X_test = test_df[feature_cols]
            y_test = test_df['churn']

            # Build and evaluate the model
            build_model(X_train, X_test, y_train, y_test)
        else:
            print("No customer data available for modeling.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 
