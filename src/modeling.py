import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import logging

def train_seasonal_naive(train_series: pd.Series, H: int = 24) -> pd.DataFrame:
    """
    Generates a 24-hour forecast using the value from the same hour previous day.
    """
    logging.info("Training Seasonal Naive baseline...")
    # Get the last 24 hours of the training series
    if len(train_series) < 24:
        raise ValueError("Not enough data for seasonal naive model (need at least 24 hours).")
        
    forecast_values = train_series.iloc[-24:].values
    
    # Create forecast DataFrame
    forecast_index = pd.date_range(start=train_series.index.max() + pd.Timedelta(hours=1), periods=H, freq='h')
    df_forecast = pd.DataFrame(forecast_values, index=forecast_index, columns=['yhat_naive'])
    logging.info("Seasonal Naive forecast generated.")
    return df_forecast

def train_ml_model(X_train: pd.DataFrame, y_train: pd.Series, model_type='ridge'):
    """
    Trains a simple ML model (Ridge or LightGBM).
    """
    logging.info(f"Training {model_type} model...")
    
    if model_type == 'ridge':
        
        # --- THIS IS THE NEW SAFETY CHECK ---
        n_samples = len(X_train)
        # Set cv to 3, but no higher than the number of samples
        n_cv = min(n_samples, 3) 
        
        # If we have less than 2 samples, CV is impossible.
        if n_cv < 2:
            n_cv = None # Use default leave-one-out cross-validation
            
        logging.info(f"Using n_cv={n_cv} for RidgeCV (n_samples={n_samples})")
        
        model = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=n_cv)
        # ------------------------------------
        
        model.fit(X_train, y_train)
        logging.info(f"Ridge model trained. Best alpha: {model.alpha_}")
        
    elif model_type == 'lightgbm':
        # LightGBM with simple, overfitting-guarded params
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=3,         # Shallow depth
            learning_rate=0.1,
            num_leaves=8,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        logging.info("LightGBM model trained.")
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    return model

def predict_with_ml_model(model, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a 24-hour forecast from a trained ML model.
    """
    yhat = model.predict(X_test)
    
    # Ensure forecast is not negative (demand can't be negative)
    yhat[yhat < 0] = 0
    
    df_forecast = pd.DataFrame(yhat, index=X_test.index, columns=['yhat_ml'])
    logging.info("ML model forecast generated.")
    return df_forecast