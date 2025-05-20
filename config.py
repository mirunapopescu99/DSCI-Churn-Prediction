DATA_PATH = "online_retail_II.csv"
CUTOFF_DATE = "2011-09-01"
CHURN_WINDOW = 180

# Model hyperparameters for grid search
MODEL_PARAMS = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Train/test split configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation folds
CV_FOLDS = 5 
