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

def create_features(df):
    df = df.copy()  # Avoid SettingWithCopyWarning

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
    print(f"InvoiceDate dtype after conversion: {df['invoicedate'].dtype}")

    df = df.dropna(subset=['invoicedate'])

    if not pd.api.types.is_datetime64_any_dtype(df['invoicedate']):
        print("❌ Error: invoicedate is not datetime!")
        return None

    cutoff_date = pd.to_datetime("2011-12-10")
    df['recency'] = (cutoff_date - df['invoicedate']).dt.days

    customer_data = df.groupby('customer_id').agg({
        'recency': 'min',
        'invoice': pd.Series.nunique,
        'price': 'sum',
        'stockcode': 'nunique'
    }).reset_index()

    customer_data.rename(columns={
        'invoice': 'frequency',
        'price': 'monetary',
        'stockcode': 'product_diversity'
    }, inplace=True) 

    customer_data['average_spend'] = customer_data['monetary'] / customer_data['frequency']
    customer_data['churn'] = np.where(customer_data['recency'] > 180, 1, 0)

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


