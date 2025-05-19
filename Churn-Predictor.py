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
    df.fillna(0, inplace=True)
    df = df[df['Quantity'] > 0]

    print(f"Cleaned DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def create_features(df, cutoff_date_str='2011-09-01', churn_window=180):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
    df = df.dropna(subset=['invoicedate'])

    # Set cutoff to the actual last date in the dataset
    cutoff_date = pd.to_datetime(cutoff_date_str) 


    # Split into pre-cutoff and post-cutoff
    df_before = df[df['invoicedate'] < cutoff_date]
    df_after = df[df['invoicedate'] >= cutoff_date]


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

    features['monetary'] = df.groupby('customer_id').apply(lambda x: (x['quantity'] * x['price']).sum(), include_groups=False)
    
    features['average_spend'] = features['monetary'] / features['frequency'].replace(0, np.nan)
    features['average_spend'] = features['average_spend'].fillna(0)

    last_dates = df_after.groupby('customer_id')['invoicedate'].min()
    reactivation_days = (last_dates - cutoff_date).dt.days

    # Initialize churn labels: default to 1 (churned)
    features['churn'] = 1

    # For customers who returned within churn_window, set churn = 0
    for cust_id, days in reactivation_days.items():
        if cust_id in features.index and days <= churn_window:
            features.at[cust_id, 'churn'] = 0 

    print("Features Created:", features.shape[1], "columns")
    print(features['churn'].value_counts())

    return features


def build_model(train_df, test_df): 
    """
    Build and train a Random Forest model to predict customer churn.
    """
    feature_cols = ['recency', 'frequency', 'monetary', 'average_spend']
    X_train = train_df[feature_cols]
    y_train = train_df['churn']

    X_test = test_df[feature_cols]
    y_test = test_df['churn']


    numerical_features = feature_cols
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

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

    cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
    print(f"Cross-validated Accuracy: {cv_scores.mean():.4f}")

    ConfusionMatrixDisplay.from_estimator(grid_search.best_estimator_, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.show()

# Feature importance
    best_model = grid_search.best_estimator_
    classifier = best_model.named_steps['classifier']
    importances = classifier.feature_importances_

    feat_importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(feat_importances['feature'], feat_importances['importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    df = load_data()

    cleaned_df = clean_data(df)

    customer_data = create_features(cleaned_df)

    build_model(customer_data)


