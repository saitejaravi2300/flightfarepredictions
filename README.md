# Flight Fare Prediction

This project uses the Excel files in `datasets/` to build a machine learning pipeline for flight fare prediction.

## Files
- `datasets/Data_Train.xlsx` – training data with Price
- `datasets/Test_set.xlsx` – test data for prediction
- `notebooks/Flight_Fare_Prediction.ipynb` – end-to-end notebook workflow
- `src/` – preprocessing, training, evaluation, and prediction scripts
- `models/` – saved model and metrics

## Quick start
1. Install dependencies: `pip install -r requirements.txt`
2. Train models: `python src/model_evaluation.py`
3. Generate predictions: `python src/prediction.py`
