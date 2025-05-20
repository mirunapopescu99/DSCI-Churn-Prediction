import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by removing duplicates, handling missing values, and filtering out negative or zero quantities.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw retail dataset.

    Returns:
    --------
    pd.DataFrame
        Cleaned dataset ready for feature engineering.

    Raises:
    -------
    ValueError
        If the input DataFrame is None or empty.
    """
    if df is None or df.empty:
        raise ValueError("Empty DataFrame received for cleaning.")

    print("Cleaning data...")
    df = df.drop_duplicates()
    df = df.fillna(0)
    df = df[df['Quantity'] > 0]

    print(f"Cleaned data shape: {df.shape}")
    return df 
