import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

def calculate_metrics(y_true: np.array, y_pred: np.array) -> dict:
    """
    Calculates MAE, WAPE, and sMAPE.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Handle potential division by zero
    epsilon = 1e-10
    
    mae = np.mean(np.abs(y_true - y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + epsilon)
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))
    
    return {
        "MAE": mae,
        "WAPE_Percent": wape * 100,
        "sMAPE_Percent": smape * 100
    }

def plot_forecast_vs_actuals(history: pd.Series, forecast: pd.DataFrame, actuals: pd.Series, city: str, save_path: str):
    """
    Plots the last 3 days of actuals with the 24-hour forecast overlay.
    """
    logging.info("Generating forecast vs. actuals plot...")
    plt.figure(figsize=(15, 7))
    
    plt.plot(history.index, history.values, label='Actuals (History)', color='blue')
    plt.plot(actuals.index, actuals.values, label='Actuals (Forecast Period)', color='orange')
    
    for col in forecast.columns:
        if 'yhat' in col:
            plt.plot(forecast.index, forecast[col], label=f'Forecast ({col})', linestyle='--')
            
    plt.title(f"24-Hour Demand Forecast vs. Actuals - {city} (Last 3 Days + Forecast)")
    plt.xlabel("Timestamp (IST)")
    plt.ylabel("Demand (hourly_kwh)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logging.info(f"Plot saved to {save_path}")
    plt.close()

def plot_horizon_mae(y_true: pd.Series, y_pred: pd.Series, model_name: str, city: str, save_path: str):
    """
    Plots the Mean Absolute Error for each hour of the forecast horizon.
    """
    logging.info("Generating horizon-wise MAE plot...")
    errors = np.abs(y_true - y_pred)
    # Calculate horizon number (1 to 24)
    horizon_hours = (errors.index.hour - errors.index.hour.min()) % 24 + 1
    
    mae_by_horizon = errors.groupby(horizon_hours).mean()
    
    plt.figure(figsize=(12, 6))
    mae_by_horizon.plot(kind='bar', zorder=3)
    
    plt.title(f"Horizon-wise MAE for {model_name} - {city}")
    plt.xlabel("Forecast Horizon (Hour)")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logging.info(f"Plot saved to {save_path}")
    plt.close()

def save_artifacts(forecast_df, metrics_df, artifacts_path):
    """
    Saves the forecast CSV and metrics CSV.
    """
    os.makedirs(artifacts_path, exist_ok=True)
    
    # Save forecast
    forecast_path = os.path.join(artifacts_path, f"forecast_T_plus_24.csv")
    forecast_df.to_csv(forecast_path)
    logging.info(f"Forecast CSV saved to {forecast_path}")
    
    # Save metrics
    metrics_path = os.path.join(artifacts_path, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=True)
    logging.info(f"Metrics CSV saved to {metrics_path}")