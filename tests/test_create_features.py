import pandas as pd
from churn_predictor import create_features  # <-- replace with your module

def test_create_features_output():
    # Sample data
    data = pd.DataFrame({
        'Invoice': ['INV1', 'INV2', 'INV3'],
        'StockCode': ['A', 'B', 'C'],
        'Description': ['Desc1', 'Desc2', 'Desc3'],
        'Quantity': [1, 2, 1],
        'InvoiceDate': ['2021-01-01', '2021-01-05', '2021-01-10'],
        'Price': [10, 20, 30],
        'Customer ID': [1, 1, 2]
    })

    # Convert dates, run feature creation
    features = create_features(data, cutoff_date_str='2021-01-15', churn_window=7)

    # Check if features contain expected columns
    expected_cols = ['recency', 'frequency', 'monetary', 'average_spend', 'churn']
    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"

    # Check values
    assert len(features) >= 2, "Should have at least two customers"

if __name__ == "__main__":
    test_create_features_output()
    print("All `create_features()` tests passed!") 
