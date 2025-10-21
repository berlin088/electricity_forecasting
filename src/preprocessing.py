import pandas as pd
import numpy as np
import logging

def resample_to_hourly(df: pd.DataFrame, target_col='Usage (kwh)') -> pd.DataFrame:
    """
    Resamples the 3-minute data to hourly data by summing.
    Ensures a continuous timestamp index in IST.
    """
    # Convert index to IST
    try:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    except TypeError:
        # Already localized, just convert
        df.index = df.index.tz_convert('Asia/Kolkata')
    except Exception as e:
        logging.warning(f"Timestamp localization warning: {e}. Assuming index is already IST.")
        df.index = df.index.tz_localize(None).tz_localize('Asia/Kolkata')

    
    # Resample by summing
    df_hourly = df[target_col].resample('h').sum().to_frame()
    df_hourly = df_hourly.rename(columns={target_col: 'hourly_kwh'})
    
    # Create a continuous hourly index
    full_index = pd.date_range(start=df_hourly.index.min(), end=df_hourly.index.max(), freq='h', tz='Asia/Kolkata')
    df_hourly = df_hourly.reindex(full_index)
    
    logging.info(f"Resampled to {len(df_hourly)} hourly records.")
    return df_hourly

def impute_gaps(df: pd.DataFrame, target_col='hourly_kwh', method='linear') -> pd.DataFrame:
    """
    Handles small gaps with conservative imputation (linear).
    """
    nan_count_before = df[target_col].isna().sum()
    if nan_count_before > 0:
        df[target_col] = df[target_col].interpolate(method=method, limit_direction='both')
        nan_count_after = df[target_col].isna().sum()
        imputed_count = nan_count_before - nan_count_after
        logging.info(f"Imputed {imputed_count} missing values using '{method}' interpolation. {nan_count_after} NaNs remain.")
        
        # If any NaNs remain (e.g., at the very start), fill with 0
        if nan_count_after > 0:
            df[target_col] = df[target_col].fillna(0)
            logging.warning(f"Filled {nan_count_after} remaining NaNs with 0.")
            
    return df

def cap_outliers(df: pd.DataFrame, target_col='hourly_kwh', lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """
    Caps extreme outliers using a percentile-based rule.
    """
    lower_limit = df[target_col].quantile(lower_q)
    upper_limit = df[target_col].quantile(upper_q)
    
    # Audit trail
    logging.info(f"Outlier Capping: Lower limit ({lower_q*100}%) = {lower_limit:.2f}, Upper limit ({upper_q*100}%) = {upper_limit:.2f}")
    
    capped_lower = (df[target_col] < lower_limit).sum()
    capped_upper = (df[target_col] > upper_limit).sum()
    
    logging.info(f"Capping {capped_lower} values below lower limit and {capped_upper} values above upper limit.")
    
    df[target_col] = df[target_col].clip(lower_limit, upper_limit)
    return df

def run_preprocessing(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline.
    """
    logging.info("Starting preprocessing...")
    df_hourly = resample_to_hourly(raw_df)
    df_imputed = impute_gaps(df_hourly)
    df_cleaned = cap_outliers(df_imputed)
    logging.info("Preprocessing complete.")
    return df_cleaned