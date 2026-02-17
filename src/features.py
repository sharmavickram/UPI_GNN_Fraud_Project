# features.py is responsible for converting raw UPI transaction logs into numerical tensors that 
# represent "fraud signals". Because GNNs learn from both node attributes and connectivity, we must 
# engineer features that capture individual behavior and temporal patterns. 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_upi_data(df, scaler=None, feature_list=None):
    print("🛠️  Running high-signal feature engineering...")

    # 1. Standardize internal names to lowercase
    df.columns = df.columns.str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Map day_of_week
    day_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 
               'friday': 4, 'saturday': 5, 'sunday': 6}
    if df['day_of_week'].dtype == 'object':
        df['day_of_week'] = df['day_of_week'].str.lower().map(day_map)

    # 3. Behavioral Velocity
    df = df.sort_values(['sender_bank', 'timestamp']).reset_index(drop=True)
    df['tx_velocity_1h'] = (
        df.set_index('timestamp')
        .groupby('sender_bank')['amount (inr)']
        .rolling('1h')
        .count()
        .values
    )

    # 4. Add Time-based numeric features
    if 'hour_of_day' not in df.columns:
        df['hour_of_day'] = df['timestamp'].dt.hour

    # 5. One-Hot Encoding for categorical signals
    # We use 'device_type' and 'merchant_category'
    df = pd.get_dummies(df, columns=['device_type', 'merchant_category'], prefix=['dev', 'cat'])

    # 6. Define feature sets (FIX: Use lowercase 'amount (inr)')
    binary_cols = [col for col in df.columns if col.startswith(('dev_', 'cat_'))]
    numeric_cols = ['amount (inr)', 'tx_velocity_1h', 'hour_of_day', 'day_of_week']
    all_features = numeric_cols + binary_cols

    # 7. Scaling
    if scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = df[numeric_cols].fillna(0)
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = df[numeric_cols].fillna(0)
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Ensure all expected columns exist
    if feature_list is not None:
        for col in feature_list:
            if col not in df.columns:
                df[col] = 0
        all_features = feature_list
    else:
        all_features = numeric_cols + binary_cols

    return df, scaler, all_features