import numpy as np
import pandas as pd


TRAIN_PATH = 'datasets/Data_Train.xlsx'
TEST_PATH = 'datasets/Test_set.xlsx'


def parse_duration(value):
    if pd.isna(value):
        return np.nan
    text = str(value).replace(' ', '')
    if 'h' in text and 'm' in text:
        h, m = text.split('h')
        m = m.replace('m', '')
        return int(h) * 60 + int(m)
    if 'h' in text:
        return int(text.replace('h', '').replace('m', '')) * 60
    if 'm' in text:
        return int(text.replace('m', ''))
    return np.nan


def load_datasets(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    return train_df, test_df


def clean_text_columns(train_df, test_df):
    for col in ['Airline', 'Source', 'Destination', 'Route', 'Additional_Info']:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str).str.strip()
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str).str.strip()
    return train_df, test_df


def normalize_datasets(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df, test_df = clean_text_columns(train_df, test_df)

    # Convert date/time fields
    train_df['Date_of_Journey'] = pd.to_datetime(train_df['Date_of_Journey'], format='%d/%m/%Y', errors='coerce', dayfirst=True)
    test_df['Date_of_Journey'] = pd.to_datetime(test_df['Date_of_Journey'], format='%d/%m/%Y', errors='coerce', dayfirst=True)

    train_df['Dep_Time'] = pd.to_datetime(train_df['Dep_Time'], format='mixed', errors='coerce')
    test_df['Dep_Time'] = pd.to_datetime(test_df['Dep_Time'], format='mixed', errors='coerce')

    train_df['Arrival_Time'] = pd.to_datetime(train_df['Arrival_Time'], format='mixed', errors='coerce')
    test_df['Arrival_Time'] = pd.to_datetime(test_df['Arrival_Time'], format='mixed', errors='coerce')

    # Duration in minutes
    train_df['Duration_Minutes'] = train_df['Duration'].apply(parse_duration)
    test_df['Duration_Minutes'] = test_df['Duration'].apply(parse_duration)

    # Parse stops as numeric
    train_df['Total_Stops'] = (
        train_df['Total_Stops']
        .replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
        .astype('Int64')
    )
    test_df['Total_Stops'] = (
        test_df['Total_Stops']
        .replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
        .astype('Int64')
    )

    return train_df, test_df


def build_features(df):
    x_df = df.copy()
    x_df['Dep_Hour'] = x_df['Dep_Time'].dt.hour
    x_df['Arr_Hour'] = x_df['Arrival_Time'].dt.hour
    x_df['Journey_Month'] = x_df['Date_of_Journey'].dt.month
    x_df['Journey_Weekday'] = x_df['Date_of_Journey'].dt.dayofweek
    x_df['Is_Morning_Flight'] = (x_df['Dep_Hour'] < 12).astype(int)
    return x_df
