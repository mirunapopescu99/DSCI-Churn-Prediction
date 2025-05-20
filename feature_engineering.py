import pandas as pd
import numpy as np

def create_features(
    df: pd.DataFrame, 
    cutoff_date_str: str = '2011-09-01', 
    churn_window: int = 180
) -> pd.DataFrame:
    """
    Create customer-level features and churn labels based on transaction history.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned retail transaction data.
    cutoff_date_str : str
        Cutoff date to split history and define churn window.
    churn_window : int
        Number of days after cutoff to consider a customer as active (not churned).

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by customer_id with features and churn labels.
    """

    df = df.copy()

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Ensure necessary columns exist
    required_cols = {'invoicedate', 'customer_id', 'invoice', 'quantity', 'price', 'description'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Parse dates and drop rows with invalid invoicedate
    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
    df = df.dropna(subset=['invoicedate'])

    cutoff_date = pd.to_datetime(cutoff_date_str)

    # Split data into before and after cutoff date
    df_before = df[df['invoicedate'] < cutoff_date]
    df_after = df[df['invoicedate'] >= cutoff_date]

    # Aggregate features from pre-cutoff transactions
    features = df_before.groupby('customer_id').agg({
        'invoicedate': lambda x: (cutoff_date - x.max()).days,
        'invoice': 'nunique',
        'quantity': 'sum',
        'price': 'mean',
        'description': 'count',
    }).rename(columns={
        'invoicedate': 'recency',
        'invoice': 'frequency',
        'quantity': 'total_quantity',
        'price': 'avg_price',
        'description': 'total_items'
    })

    # Calculate monetary value using vectorized operation
    monetary = df_before.assign(total=df_before['quantity'] * df_before['price']) \
                       .groupby('customer_id')['total'].sum()
    features['monetary'] = monetary

    # Average spend per invoice (frequency)
    features['average_spend'] = features['monetary'] / features['frequency'].replace(0, np.nan)
    features['average_spend'].fillna(0, inplace=True)

    # Calculate churn labels
    last_dates = df_after.groupby('customer_id')['invoicedate'].min()
    reactivation_days = (last_dates - cutoff_date).dt.days

    features['churn'] = 1  # Default churned

    # Mark customers as active if they purchased within churn window
    active_customers = reactivation_days[reactivation_days <= churn_window].index
    features.loc[active_customers, 'churn'] = 0

    return features 
