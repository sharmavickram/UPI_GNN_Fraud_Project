# features.py is responsible for converting raw UPI transaction logs into numerical tensors that 
# represent "fraud signals". Because GNNs learn from both node attributes and connectivity, we must 
# engineer features that capture individual behavior and temporal patterns. 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_upi_data(df):
    print("🛠️  Running high-signal feature engineering...")
    
    # 1. Clean column names and convert timestamp
    df.columns = df.columns.str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Map day_of_week to numbers
    day_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    if df['day_of_week'].dtype == 'object':
        df['day_of_week'] = df['day_of_week'].map(day_map)

    # 3. Calculate Behavioral Velocity (CRITICAL FIX: Define before scaling)
    df = df.sort_values(['sender_bank', 'timestamp']).reset_index(drop=True)
    df['tx_velocity_1h'] = (
        df.set_index('timestamp')
        .groupby('sender_bank')['amount (INR)']
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

    # 6. Define feature sets
    binary_cols = [col for col in df.columns if col.startswith(('dev_', 'cat_'))]
    numeric_cols = ['amount (INR)', 'tx_velocity_1h', 'hour_of_day', 'day_of_week']
    all_features = numeric_cols + binary_cols

    # 7. Scaling numeric features only
    scaler = StandardScaler()
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Fill NAs for binary columns just in case
    df[binary_cols] = df[binary_cols].fillna(0).astype(int)
    
    print(f"📊 Feature Engineering Complete. Total features: {len(all_features)}")
    return df, scaler, all_features