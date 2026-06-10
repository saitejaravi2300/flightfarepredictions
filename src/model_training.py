import os
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_preprocessing import build_features, load_datasets, normalize_datasets


FEATURE_COLUMNS = [
    'Airline', 'Date_of_Journey', 'Source', 'Destination', 'Total_Stops',
    'Dep_Time', 'Arrival_Time', 'Duration_Minutes', 'Dep_Hour', 'Arr_Hour',
    'Journey_Month', 'Journey_Weekday'
]


def build_preprocessor():
    numeric_features = ['Duration_Minutes', 'Dep_Hour', 'Arr_Hour', 'Journey_Month', 'Journey_Weekday', 'Total_Stops']
    categorical_features = ['Airline', 'Source', 'Destination']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
        ]
    )
    return preprocessor


def train_models():
    train_df, _ = load_datasets()
    train_df, _ = normalize_datasets(train_df, _)
    train_df = build_features(train_df)

    X = train_df[FEATURE_COLUMNS].copy()
    y = train_df['Price']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=250, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    }

    preprocessor = build_preprocessor()
    results = {}

    for name, model in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        mse = mean_squared_error(y_val, pred)
        results[name] = {
            'mae': mean_absolute_error(y_val, pred),
            'rmse': np.sqrt(mse),
            'r2': r2_score(y_val, pred),
        }

    best_name = max(results, key=lambda k: results[k]['r2'])
    best_model = models[best_name]
    final_pipe = Pipeline([('preprocessor', preprocessor), ('model', best_model)])
    final_pipe.fit(X, y)

    os.makedirs('models', exist_ok=True)
    joblib.dump(final_pipe, 'models/best_model.pkl')

    return results, best_name, final_pipe
