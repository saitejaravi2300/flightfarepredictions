import os
import sys

import joblib
import pandas as pd
from flask import Flask, render_template, request

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_preprocessing import build_features, normalize_datasets

app = Flask(__name__)

MODEL_PATH = os.path.join('models', 'best_model.pkl')
model = joblib.load(MODEL_PATH)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        input_df = pd.DataFrame([{
            'Airline': request.form['airline'],
            'Date_of_Journey': request.form['date_of_journey'],
            'Source': request.form['source'],
            'Destination': request.form['destination'],
            'Total_Stops': request.form['total_stops'],
            'Dep_Time': request.form['dep_time'],
            'Arrival_Time': request.form['arrival_time'],
            'Duration': request.form['duration'],
            'Route': request.form.get('route', ''),
            'Additional_Info': request.form.get('additional_info', '')
        }])

        input_df, _ = normalize_datasets(input_df, input_df.iloc[0:0])
        input_df = build_features(input_df)

        feature_columns = [
            'Airline', 'Date_of_Journey', 'Source', 'Destination', 'Total_Stops',
            'Dep_Time', 'Arrival_Time', 'Duration_Minutes', 'Dep_Hour', 'Arr_Hour',
            'Journey_Month', 'Journey_Weekday'
        ]
        prediction = round(float(model.predict(input_df[feature_columns])[0]), 2)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
