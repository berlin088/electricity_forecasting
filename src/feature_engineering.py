import pandas as pd
import numpy as np
import logging

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates hour-of-day (sin/cos) and day-of-week features.
    """
    df_feat = df.copy()
    idx = df_feat.index
    df_feat['hour'] = idx.hour
    df_feat['dayofweek'] = idx.dayofweek
    
    # Sin/Cos transformation for cyclical 'hour'
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    
    # One-hot encode day of week
    df_feat = pd.get_dummies(df_feat, columns=['dayofweek'], drop_first=True)
    
    return df_feat.drop(columns=['hour'])

def create_lag_features(df: pd.DataFrame, target_col='hourly_kwh') -> pd.DataFrame:
    """
    Creates lagged features as specified.
    """
    df_feat = df.copy()
    
    lags = [24, 25, 26, 48, 168] # 1 day, 1 day + 1hr, 1 day + 2hr, 2 days, 1 week
    for lag in lags:
        df_feat[f'lag_{lag}'] = df_feat[target_col].shift(lag)
        
    # 24-hour rolling mean, also lagged by 24 hours
    df_feat['rolling_mean_24_lag24'] = df_feat[target_col].shift(24).rolling(window=24).mean()
    
    return df_feat

def merge_weather_features(df_feat: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges weather data (historical or forecast) onto the feature DataFrame.
    """
    if weather_df is not None:
        logging.info("Merging weather features...")
        # Ensure both are in the same timezone (IST)
        if df_feat.index.tz != weather_df.index.tz:
             weather_df.index = weather_df.index.tz_convert(df_feat.index.tz)
             
        df_feat = df_feat.join(weather_df)
        
        nan_before = df_feat['temperature'].isna().sum()
        if nan_before > 0:
            logging.info(f"Weather data has {nan_before} NaNs before imputation (likely from join mismatch).")
        
        # Impute any missing weather data (e.g., from API issues)
        df_feat['temperature'] = df_feat['temperature'].interpolate(method='linear', limit_direction='both')
        
        # --- NEW SAFETY NET ---
        # If any NaNs remain (e.g., join failed and interpolate couldn't fix it), fill with 0
        nan_after = df_feat['temperature'].isna().sum()
        if nan_after > 0:
            logging.warning(f"{nan_after} NaNs remain after interpolation. Filling with 0.")
            df_feat['temperature'] = df_feat['temperature'].fillna(0)
        # --------------------
            
    else:
        # If no weather is used, create a dummy column of 0s
        # This prevents the 'temperature' column from being missing later
        logging.info("No weather data provided. Creating dummy 'temperature' column with 0s.")
        df_feat['temperature'] = 0
    
    return df_feat

def create_features(df: pd.DataFrame, weather_df: pd.DataFrame = None, target_col='hourly_kwh') -> pd.DataFrame:
    """
    Runs the full feature engineering pipeline.
    """
    df_feat = create_time_features(df)
    df_feat = create_lag_features(df_feat, target_col)
    df_feat = merge_weather_features(df_feat, weather_df)
    
    # Add the target column back for easy splitting
    df_feat['target'] = df[target_col]
    
    return df_feat