import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath, input_columns, target_column, test_storm_ids):
    df = pd.read_csv(filepath, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)

    test_df = df[df['StormID'].isin(test_storm_ids)]
    train_df = df[~df['StormID'].isin(test_storm_ids)]

    X_train = train_df[input_columns]
    y_train = train_df[target_column]
    X_test = test_df[input_columns]
    y_test = test_df[target_column]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test
