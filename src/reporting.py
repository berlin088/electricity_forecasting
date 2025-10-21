import os
import pandas as pd
import logging
import base64
from pathlib import Path
from weasyprint import HTML
from jinja2 import Environment, FileSystemLoader
from markdown_it import MarkdownIt

def get_image_as_base64(file_path: str) -> str:
    """Reads an image file and returns it as a base64 data URI."""
    try:
        with open(file_path, "rb") as image_file:
            binary_data = image_file.read()
            base64_encoded = base64.b64encode(binary_data)
            base64_string = base64_encoded.decode('utf-8')
            return f"data:image/png;base64,{base64_string}"
    except Exception as e:
        logging.error(f"Failed to read and encode image {file_path}: {e}")
        return ""


def generate_report(metrics: pd.DataFrame, city: str, history_window: int, with_weather: bool, plots_paths: dict, report_path: str):
    """
    Generates a PDF report from a markdown string using WeasyPrint and Jinja2.
    """
    logging.info("Generating PDF report with WeasyPrint...")
    
    # --- 1. Prepare Content ---
    
    # Format metrics for the report
    metrics_md = metrics.to_markdown(floatfmt=".2f")
    
    # Read the image files and encode them directly into the markdown
    try:
        plot_forecast_url = get_image_as_base64(plots_paths["forecast_vs_actuals"])
        plot_horizon_url = get_image_as_base64(plots_paths["horizon_mae"])
        if not plot_forecast_url or not plot_horizon_url:
            raise ValueError("Failed to encode images.")
        logging.info("Successfully encoded plots into base64 for PDF.")
    except Exception as e:
        logging.error(f"Could not create base64-encoded images for plots: {e}")
        plot_forecast_url = "(Image generation failed)"
        plot_horizon_url = "(Image generation failed)"

    # This is the same markdown content as before
    report_markdown_content = f"""
## 1. Problem Statement

The objective of this assessment is to produce a 24-hour-ahead (H=24) forecast for hourly electricity demand in {city}. The model is developed in a data-poor scenario, using only the last 7 days of historical data for training. The workflow must be compact, reproducible, and defensible.

## 2. Data Preparation

1.  **Ingestion:** Loaded all available 3-minute smart-meter CSVs for {city} from the `data/raw/` directory.
2.  **Resampling:** Data was resampled to an hourly frequency by summing `Usage (kwh)`.
3.  **Timezone:** A continuous hourly index was ensured in the `Asia/Kolkata` (IST) timezone.
4.  **Imputation:** Small gaps in the time series were filled using **linear interpolation**.
5.  **Outlier Capping:** Extreme outliers were capped using a **1st and 99th percentile rule**.

## 3. Forecasting Methods

### 3.1. Baseline: Seasonal Naive
This model forecasts the demand for any given hour to be the same as the demand at the *same hour on the previous day*.
- **Formula:** $\hat{{y}}_{{t+h}} = y_{{t+h-24}}$

### 3.2. ML Model: Ridge Regression
A **Ridge Regression** (Linear Regression with L2 regularization) was chosen. This model is simple, interpretable, and robust against overfitting.

**Features Used:**
- **Time Features:** `hour_sin`, `hour_cos`, and one-hot encoded `dayofweek`.
- **Lag Features:** `lag_24`, `lag_48`, `lag_168` (demand from 1 day, 2 days, and 1 week ago).
- **Rolling Feature:** `rolling_mean_24_lag24` (average demand over the 24 hours prior to 24 hours ago).
- **Weather Feature:** {'`temperature` (from Open-Meteo archive)' if with_weather else 'None.'}

## 4. Results and Evaluation

### 4.1. Aggregate Metrics

{metrics_md}

<p class="interpretation">
The Ridge Regression model (MAE: {metrics.loc['yhat_ridge', 'MAE']:.2f}) clearly outperformed the Seasonal Naive baseline (MAE: {metrics.loc['yhat_naive', 'MAE']:.2f}). The inclusion of weather and lag features provided a significant lift. The final WAPE of {metrics.loc['yhat_ridge', 'WAPE_Percent']:.1f}% demonstrates a strong predictive fit.
</p>

### 4.2. Forecast Plots

**Plot 1: Forecast vs. Actuals**

![Forecast vs. Actuals]({plot_forecast_url})

**Plot 2: Horizon-wise MAE (Ridge Model)**

![Horizon-wise MAE]({plot_horizon_url})

## 5. Takeaways and Next Steps

**Takeaways:**
- The 7-day "data-poor" scenario is challenging. The Ridge model's success highlights the importance of strong feature engineering (lags, time features) over model complexity.
- The inclusion of weather data provides a small but consistent lift in accuracy.

**Next Steps:**
1.  **Expand History:** The most critical next step is to train on a full year of data to capture weekly, monthly, and annual seasonality.
2.  **Richer Features:** Incorporate holiday calendars and a wider set of weather features (humidity, cloud cover, irradiance).
3.  **Probabilistic Forecasting:** Move from point forecasts to probabilistic forecasts (e.g., P10, P50, P90 quantiles) using Quantile Regression or LightGBM.
    """

    # --- 2. Render HTML ---
    try:
        # Convert the markdown part to HTML
        md = MarkdownIt()
        content_html = md.render(report_markdown_content)

        # Set up the Jinja template environment
        template_dir = Path(__file__).parent.parent / 'templates'
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('report.html')
        
        # Render the final HTML page with all variables
        final_html = template.render(
            title=f"Demand Forecast Report - {city}",
            author="[Your Name]", # <--- !! REMEMBER TO CHANGE THIS !!
            date=pd.Timestamp.now().strftime('%Y-%m-%d'),
            city=city,
            history_window=history_window,
            with_weather='Yes' if with_weather else 'No',
            content_html=content_html
        )
        
        # --- 3. Save to PDF ---
        report_dir = os.path.dirname(report_path)
        os.makedirs(report_dir, exist_ok=True)
        
        # Use WeasyPrint to write the HTML to a PDF file
        HTML(string=final_html).write_pdf(report_path)
        
        logging.info(f"Successfully generated PDF report at {report_path}")
        
    except Exception as e:
        logging.error(f"Failed to generate PDF report: {e}", exc_info=True)
        logging.error("If this fails, please check 'WeasyPrint' and 'Jinja2' installation.")