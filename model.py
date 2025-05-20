import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from config import MODEL_PARAMS, RANDOM_STATE, CV_FOLDS


def build_model(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
) -> Pipeline:
    """
    Build, train, and evaluate RandomForest model with hyperparameter tuning.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Feature matrices for train and test
    y_train, y_test : pd.Series
        Labels for train and test

    Returns:
    --------
    best_estimator_ : sklearn Pipeline
        Trained pipeline with best parameters
    """

    feature_cols = X_train.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), feature_cols)]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    param_grid = {
        'classifier__n_estimators': MODEL_PARAMS['n_estimators'],
        'classifier__max_depth': MODEL_PARAMS['max_depth'],
        'classifier__min_samples_split': MODEL_PARAMS['min_samples_split']
    }

    try:
        grid_search = GridSearchCV(pipeline, param_grid, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
        plt.title("Confusion Matrix")
        plt.show()

        # Feature importance
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

        return best_model

    except Exception as e:
        print(f"Error during model training or evaluation: {e}")
        raise e 
