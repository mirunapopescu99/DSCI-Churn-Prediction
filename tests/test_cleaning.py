import pandas as pd
from churn_predictor import clean_data  # <-- replace with your actual module name

def test_clean_removes_duplicates():
    # Create sample data with duplicates
    data = pd.DataFrame({
        'Quantity': [1, 1, 2],
        'Invoice': ['INV001', 'INV001', 'INV002'],
        'Customer ID': [123, 123, 456],
        # include other necessary fields, values don't matter much for this test
    })
    cleaned_df = clean_data(data)
    # After cleaning, should remove duplicate row with 'INV001'
    assert len(cleaned_df) == 2, f"Expected 2 rows, got {len(cleaned_df)}"

def test_clean_handles_missing():
    data = pd.DataFrame({
        'Quantity': [None, 2],
        'Invoice': ['INV003', 'INV004'],
        'Customer ID': [789, 101],
    })
    cleaned_df = clean_data(data)
    # No missing values should remain
    assert not cleaned_df.isnull().values.any(), "There are still missing values after cleaning."

if __name__ == "__main__":
    test_clean_removes_duplicates()
    test_clean_handles_missing()
    print("All `clean_data()` tests passed!") 
