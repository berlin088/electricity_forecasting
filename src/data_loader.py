import os
import glob
import pandas as pd
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lat/Lon for weather API
CITY_COORDS = {
    "Bareilly": {"latitude": 28.37, "longitude": 79.42},
    "Mathura": {"latitude": 27.49, "longitude": 77.67}
}

def load_city_data(raw_data_path: str, city_name: str) -> pd.DataFrame:
    """
    Loads all CSV files for a given city from the raw data path,
    concatenates them, and returns a single sorted DataFrame.
    """
    search_pattern = os.path.join(raw_data_path, f"*{city_name}*.csv")
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        logging.error(f"No data files found for city '{city_name}' in '{raw_data_path}'")
        raise FileNotFoundError(f"No data files found for {city_name}")

    logging.info(f"Found {len(file_list)} files for {city_name}. Loading...")
    
    df_list = []
    for file in file_list:
        try:
            # Read as string first to avoid pandas auto-parsing
            df = pd.read_csv(file, dtype={'x_Timestamp': str})
            df_list.append(df)
        except Exception as e:
            logging.warning(f"Could not load file {file}: {e}")
            
    if not df_list:
        logging.error(f"All files for {city_name} failed to load.")
        raise ValueError(f"Could not load any data for {city_name}")

    # Concatenate all dataframes
    full_df = pd.concat(df_list, ignore_index=True)
    
    actual_time_col = 'x_Timestamp'
    actual_data_col = 't_kWh'

    if actual_time_col not in full_df.columns or actual_data_col not in full_df.columns:
        logging.error(f"Required columns '{actual_time_col}' or '{actual_data_col}' not found.")
        logging.info(f"Found columns: {full_df.columns.tolist()}")
        raise ValueError("Missing required columns in source data.")
        
    logging.info(f"Renaming '{actual_time_col}' to 'DateTime' and '{actual_data_col}' to 'Usage (kwh)'.")
    
    full_df = full_df.rename(columns={
        actual_time_col: 'DateTime', 
        actual_data_col: 'Usage (kwh)'
    })

    full_df['DateTime'] = pd.to_datetime(full_df['DateTime'], format='mixed', dayfirst=True)
    
    full_df = full_df.set_index('DateTime').sort_index()
    full_df = full_df[['Usage (kwh)']]
    
    logging.info(f"Successfully loaded data for {city_name} from {full_df.index.min()} to {full_df.index.max()}")
    return full_df

def fetch_weather_data(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches hourly weather (temperature) from Open-Meteo.
    Uses the ARCHIVE API since we are forecasting on historical data.
    """
    if city not in CITY_COORDS:
        raise ValueError(f"Coordinates for city '{city}' not found.")
        
    coords = CITY_COORDS[city]
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": "temperature_2m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC"  # Explicitly ask for UTC
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Raise an error for bad responses
        data = response.json()
        
        weather_df = pd.DataFrame(data['hourly'])
        weather_df = weather_df.rename(columns={"time": "timestamp", "temperature_2m": "temperature"})
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        weather_df = weather_df.set_index('timestamp')
        
        # --- THIS IS THE FIX ---
        # First, localize the naive UTC timestamp, THEN convert to IST
        weather_df.index = weather_df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        # -----------------------
        
        logging.info(f"Successfully fetched weather data for {city} from {start_date} to {end_date}")
        return weather_df
        
    except Exception as e:
        logging.error(f"Failed to fetch weather data: {e}", exc_info=True)
        return None