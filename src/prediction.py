import joblib
import pandas as pd

from data_preprocessing import build_features, load_datasets, normalize_datasets


def predict_fares(model_path='models/best_model.pkl'):
    train_df, test_df = load_datasets()
    train_df, test_df = normalize_datasets(train_df, test_df)
    test_df = build_features(test_df)

    feature_columns = [
        'Airline', 'Date_of_Journey', 'Source', 'Destination', 'Total_Stops',
        'Dep_Time', 'Arrival_Time', 'Duration_Minutes', 'Dep_Hour', 'Arr_Hour',
        'Journey_Month', 'Journey_Weekday'
    ]

    model = joblib.load(model_path)
    predictions = model.predict(test_df[feature_columns])
    output = pd.DataFrame({
        'Flight_ID': range(1, len(predictions) + 1),
        'Predicted_Price': predictions
    })

    output.to_csv('models/test_predictions.csv', index=False)
    return output


if __name__ == '__main__':
    result = predict_fares()
    print(result.head())
