import os
import argparse
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path  # <-- IMPORT THE PATHLIB LIBRARY

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src import data_loader, preprocessing, feature_engineering, modeling, evaluation, reporting
except ImportError:
    print("Error: Could not import from 'src'. Make sure 'src' is in your Python path or run this script from the project root.")
    sys.exit(1)


# --- Configuration ---
RAW_DATA_PATH = os.path.join("data", "raw")
ARTIFACTS_PATH = os.path.join("artifacts", "fast_track")
REPORTS_PATH = os.path.join("reports")
FORECAST_HORIZON = 24  # 24 hours
# Use a buffer of 168 (for lag_168) + 48 (for rolling_mean safety) = 216 hours
FEATURE_BUFFER_HOURS = 216 # Hours needed for lags

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting forecast pipeline...")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run 24-hour electricity demand forecast.")
    parser.add_argument("--city", type=str, required=True, choices=["Bareilly", "Mathura"], help="City to forecast.")
    parser.add_argument("--history_window_days", type=int, default=7, help="Number of days for training history.")
    parser.add_argument("--with_weather", type=lambda x: (str(x).lower() == 'true'), default=True, help="Set to 'true' to use weather data.")
    parser.add_argument("--make_plots", type=lambda x: (str(x).lower() == 'true'), default=True, help="Set to 'true' to generate plots.")
    parser.add_argument("--save_report", type=lambda x: (str(x).lower() == 'true'), default=True, help="Set to 'true' to generate PDF report.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    logging.info(f"Running pipeline with args: {args}")

    try:
        # 1. Load Data
        raw_df = data_loader.load_city_data(RAW_DATA_PATH, args.city)

        # 2. Preprocessing
        df_clean = preprocessing.run_preprocessing(raw_df)

        # 3. Define Train/Test Windows
        if len(df_clean) < (args.history_window_days * 24 + FORECAST_HORIZON + FEATURE_BUFFER_HOURS):
            logging.error(f"Not enough data. Need at least {args.history_window_days * 24 + FORECAST_HORIZON + FEATURE_BUFFER_HOURS} total hours.")
            logging.error(f"Found only {len(df_clean)} hours. Try a smaller history window.")
            return

        # Define time splits
        test_end_time = df_clean.index.max()
        test_start_time = test_end_time - pd.Timedelta(hours=FORECAST_HORIZON - 1)
        
        train_end_time = test_start_time - pd.Timedelta(hours=1)
        train_start_time = train_end_time - pd.Timedelta(days=args.history_window_days) + pd.Timedelta(hours=1)
        
        # History for plotting (last 3 days of train)
        plot_history_start = train_end_time - pd.Timedelta(days=3)
        
        buffer_start_time = train_start_time - pd.Timedelta(hours=FEATURE_BUFFER_HOURS)
        
        # Split data
        train_data = df_clean.loc[train_start_time:train_end_time]
        test_data = df_clean.loc[test_start_time:test_end_time]
        plot_history = df_clean.loc[plot_history_start:train_end_time]
        
        logging.info(f"Train window: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} hours)")
        logging.info(f"Test window:  {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} hours)")
        
        # 4. Get Weather Data
        weather_data = None
        if args.with_weather:
            weather_start = buffer_start_time.strftime('%Y-%m-%d')
            weather_end = test_end_time.strftime('%Y-%m-%d')
            logging.info(f"Fetching weather from {weather_start} to {weather_end}")
            
            weather_data = data_loader.fetch_weather_data(args.city, weather_start, weather_end)
            if weather_data is None:
                logging.warning("Failed to fetch weather. Proceeding without it.")
                args.with_weather = False

        # 5. Feature Engineering
        logging.info(f"Running feature engineering on data from {buffer_start_time} to {test_end_time}")
        full_feature_df = feature_engineering.create_features(
            df_clean.loc[buffer_start_time:test_end_time], # Use the buffer_start_time
            weather_data
        )

        # Split into X, y
        df_train_feat = full_feature_df.loc[train_start_time:train_end_time]
        df_test_feat = full_feature_df.loc[test_start_time:test_end_time]
        
        FEATURE_COLS = [col for col in df_train_feat.columns if col not in ['target', 'hourly_kwh']]
        
        X_train, y_train = df_train_feat[FEATURE_COLS], df_train_feat['target']
        X_test, y_test = df_test_feat[FEATURE_COLS], df_test_feat['target']
        
        logging.info(f"X_train shape before dropna: {X_train.shape}")
        nan_counts = X_train.isna().sum()
        if nan_counts.sum() > 0:
            logging.warning(f"NaN counts per column in X_train BEFORE dropna:\n{nan_counts[nan_counts > 0]}")
        else:
            logging.info("X_train has no NaNs before dropna.")

        # Now we drop NaNs from X_train (created by lags)
        X_train = X_train.dropna()
        # And align y_train to match the valid (non-NaN) rows
        y_train = y_train.loc[X_train.index]
        
        logging.info(f"Training with {len(X_train)} samples after dropping NaNs.")
        
        if len(X_train) == 0:
            logging.error("X_train is empty after processing. Cannot train model.")
            raise ValueError("No training data available after feature engineering.")

        # 6. Modeling
        # 6.1. Baseline: Seasonal Naive
        df_naive_forecast = modeling.train_seasonal_naive(train_data['hourly_kwh'], H=FORECAST_HORIZON)
        
        # 6.2. ML Model: Ridge
        model_ridge = modeling.train_ml_model(X_train, y_train, model_type='ridge')
        df_ridge_forecast = modeling.predict_with_ml_model(model_ridge, X_test)

        # 7. Evaluation
        final_forecast = df_ridge_forecast.rename(columns={'yhat_ml': 'yhat_ridge'})
        final_forecast['yhat_naive'] = df_naive_forecast['yhat_naive']
        final_forecast['y_actual'] = y_test

        metrics_results = {}
        for model_name in ['yhat_ridge', 'yhat_naive']:
            metrics = evaluation.calculate_metrics(final_forecast['y_actual'], final_forecast[model_name])
            metrics_results[model_name] = metrics
            logging.info(f"Metrics for {model_name}: {metrics}")
            
        metrics_df = pd.DataFrame(metrics_results).T
        metrics_df.index.name = "Model"

        # 8. Save Artifacts
        city_artifacts_path = os.path.join(ARTIFACTS_PATH, args.city)
        os.makedirs(city_artifacts_path, exist_ok=True)
        
        evaluation.save_artifacts(final_forecast, metrics_df, city_artifacts_path)
        
        plot_paths = {}

        if args.make_plots:
            plots_dir = os.path.join(city_artifacts_path, "plots")
            
            # --- THIS IS THE FINAL FIX ---
            # Use pathlib.Path.resolve() to get a guaranteed absolute path string
            plot_path_1_obj = Path(plots_dir) / "forecast_vs_actuals.png"
            plot_path_1 = str(plot_path_1_obj.resolve())

            plot_path_2_obj = Path(plots_dir) / "horizon_mae_ridge.png"
            plot_path_2 = str(plot_path_2_obj.resolve())
            # ---------------------------

            evaluation.plot_forecast_vs_actuals(
                plot_history['hourly_kwh'], 
                final_forecast[['yhat_ridge', 'yhat_naive']], 
                y_test, 
                args.city, 
                plot_path_1 # Pass the absolute path string
            )
            plot_paths["forecast_vs_actuals"] = plot_path_1
            
            evaluation.plot_horizon_mae(y_test, final_forecast['yhat_ridge'], "Ridge", args.city, plot_path_2)
            plot_paths["horizon_mae"] = plot_path_2

        # 9. Generate Report
        if args.save_report:
            report_path = os.path.join(REPORTS_PATH, f"fast_track_report_{args.city}.pdf")
            reporting.generate_report(
                metrics_df,
                args.city,
                args.history_window_days,
                args.with_weather,
                plot_paths,
                report_path
            )
            
        logging.info("Forecast pipeline finished successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()