import pandas as pd

def load_data(path: str = "online_retail_II.csv") -> pd.DataFrame:
    """
    Loads the dataset from the specified CSV file path.

    Parameters:
    -----------
    path : str
        File path to the CSV dataset.

    Returns:
    --------
    pd.DataFrame
        Loaded dataset.

    Raises:
    -------
    FileNotFoundError
        If the specified file path does not exist.
    pd.errors.EmptyDataError
        If the file is empty.
    pd.errors.ParserError
        If the file cannot be parsed as CSV.
    """
    try:
        df = pd.read_csv(path)
        print("✅ Data Loaded Successfully")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        raise
    except pd.errors.EmptyDataError:
        print(f"❌ No data: The file at {path} is empty.")
        raise
    except pd.errors.ParserError:
        print(f"❌ Parsing error: The file at {path} could not be parsed.")
        raise 
