import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os
import schedule
import time
from datetime import datetime, timedelta, timezone

# Automatically determine start and end date for fetch
def get_date_range(parquet_file, buffer_hours=1):
    """
    Determines the date range for fetching new data.
    If parquet file exists, starts from the latest date + 1 hour.
    Otherwise, starts from 7 days ago.
    """
    if os.path.exists(parquet_file):
        try:
            existing_df = pd.read_parquet(parquet_file)
            if "date" in existing_df.columns and len(existing_df) > 0:
                latest_time = pd.to_datetime(existing_df["date"]).max()
                start_time = latest_time + timedelta(hours=1)
                print(f"📊 Found existing data up to: {latest_time}")
            else:
                start_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(days=7)
                print("📊 No date data found in existing file, starting from 7 days ago")
        except Exception as e:
            print(f"⚠️  Error reading existing file: {e}")
            start_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(days=7)
    else:
        start_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(days=7)
        print("📊 No existing file found, starting from 7 days ago")

    end_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=buffer_hours)
    
    return start_time.date().isoformat(), end_time.date().isoformat()

# -----------------------------
# Configuration
# -----------------------------
# File path configuration - modified for GitHub CI
OUTPUT_PARQUET = os.getenv("OUTPUT_FILE", "karachi_aqi_data.parquet")
DATA_DIR = os.getenv("DATA_DIR", "data")  # Changed default to 'data' directory
FULL_FILE_PATH = os.path.join(DATA_DIR, OUTPUT_PARQUET)

# Location coordinates for Karachi
LAT, LON = float(os.getenv("LATITUDE", "24.8414")), float(os.getenv("LONGITUDE", "67.1416"))

# Column order to match your existing file
COLUMN_ORDER = [
    'date', 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 
    'surface_pressure', 'wind_speed_10m', 'wind_direction_10m', 
    'precipitation', 'cloud_cover', 'pm10', 'pm2_5', 'carbon_monoxide', 
    'nitrogen_dioxide', 'sulphur_dioxide', 'ozone'
]

# -----------------------------
# Open-Meteo Client Setup
# -----------------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_aqi_data(start_date, end_date):
    """Fetch air quality data from Open-Meteo API"""
    print(f"🌬️  Fetching AQI data from {start_date} to {end_date}")
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date
    }
    
    try:
        response = openmeteo.weather_api(url, params=params)[0]
        hourly = response.Hourly()
        
        aqi_df = pd.DataFrame({
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "pm10": hourly.Variables(0).ValuesAsNumpy(),
            "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
            "carbon_monoxide": hourly.Variables(2).ValuesAsNumpy(),
            "nitrogen_dioxide": hourly.Variables(3).ValuesAsNumpy(),
            "sulphur_dioxide": hourly.Variables(4).ValuesAsNumpy(),
            "ozone": hourly.Variables(5).ValuesAsNumpy()
        })
        
        print(f"✅ AQI data fetched: {len(aqi_df)} records")
        return aqi_df
        
    except Exception as e:
        print(f"❌ Error fetching AQI data: {e}")
        raise

def fetch_weather_data(start_date, end_date):
    """Fetch weather data from Open-Meteo API"""
    print(f"🌤️  Fetching weather data from {start_date} to {end_date}")
    
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "surface_pressure", "wind_speed_10m", "wind_direction_10m",
            "precipitation", "cloud_cover"
        ]
    }
    
    try:
        response = openmeteo.weather_api(url, params=params)[0]
        hourly = response.Hourly()
        
        weather_df = pd.DataFrame({
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "dew_point_2m": hourly.Variables(2).ValuesAsNumpy(),
            "surface_pressure": hourly.Variables(3).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(4).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(5).ValuesAsNumpy(),
            "precipitation": hourly.Variables(6).ValuesAsNumpy(),
            "cloud_cover": hourly.Variables(7).ValuesAsNumpy(),
        })
        
        print(f"✅ Weather data fetched: {len(weather_df)} records")
        return weather_df
        
    except Exception as e:
        print(f"❌ Error fetching weather data: {e}")
        raise

def merge_and_update(weather_df, aqi_df, output_path):
    """Merge weather and AQI data, then update the parquet file with robust duplicate handling"""
    print("🔄 Merging weather and AQI data...")
    
    # Merge on date
    merged = pd.merge(weather_df, aqi_df, on="date", how="inner")
    
    # Reorder columns to match your existing file structure
    merged = merged[COLUMN_ORDER]
    merged = merged.sort_values("date")
    
    if len(merged) == 0:
        print("⚠️  No data to merge for the specified date range")
        return

    # Check for duplicates within the new data itself
    initial_merged_count = len(merged)
    merged = merged.drop_duplicates(subset="date", keep="last")
    if len(merged) < initial_merged_count:
        duplicates_in_new = initial_merged_count - len(merged)
        print(f"🔍 Removed {duplicates_in_new} duplicates from new data")

    if os.path.exists(output_path):
        existing = pd.read_parquet(output_path)
        existing_count = len(existing)
        
        # Find overlapping dates for detailed reporting
        existing_dates = set(existing['date'])
        new_dates = set(merged['date'])
        overlapping_dates = existing_dates.intersection(new_dates)
        
        if overlapping_dates:
            print(f"🔍 Found {len(overlapping_dates)} overlapping dates")
            print(f"📅 Overlap range: {min(overlapping_dates)} to {max(overlapping_dates)}")
        
        # Combine and remove duplicates
        combined = pd.concat([existing, merged], ignore_index=True)
        
        # Advanced duplicate handling with data validation
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset="date", keep="last")
        after_dedup = len(combined)
        
        duplicates_removed = before_dedup - after_dedup
        if duplicates_removed > 0:
            print(f"🔍 Removed {duplicates_removed} duplicate records")
        
        combined = combined.sort_values("date").reset_index(drop=True)
        
        # Ensure column order is maintained
        combined = combined[COLUMN_ORDER]
        
        # Calculate actual new records added
        actual_new_records = len(combined) - existing_count
        
        if actual_new_records > 0:
            print(f"✅ Updated {output_path}: +{actual_new_records} new records")
        else:
            print(f"📊 No new records added (all data was already present)")
            
        # Data integrity check
        if len(combined) < existing_count:
            print(f"⚠️  Warning: Combined data has fewer records than original ({len(combined)} vs {existing_count})")
            
    else:
        combined = merged
        print(f"✅ Created {output_path} with {len(combined)} records")

    # Final validation
    date_duplicates = combined['date'].duplicated().sum()
    if date_duplicates > 0:
        print(f"❌ Warning: Still have {date_duplicates} duplicate dates after processing!")
        # Remove any remaining duplicates as a safety measure
        combined = combined.drop_duplicates(subset="date", keep="last")
        print(f"🔧 Fixed remaining duplicates. Final count: {len(combined)}")

    # Save to parquet
    combined.to_parquet(output_path, index=False)
    print(f"📅 Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"📊 Total records: {len(combined)}")
    
    # Additional data quality checks
    null_dates = combined['date'].isnull().sum()
    if null_dates > 0:
        print(f"⚠️  Warning: {null_dates} records have null dates")
        
    # Check for reasonable data gaps
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values('date')
    time_diffs = combined['date'].diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=6)]
    if len(large_gaps) > 0:
        print(f"⚠️  Found {len(large_gaps)} time gaps larger than 6 hours")
        
    return combined

def validate_data_integrity(df, filename):
    """Validate data integrity and report issues"""
    print(f"\n🔍 Validating data integrity for {filename}")
    
    issues = []
    
    # Check for duplicate dates
    date_dupes = df['date'].duplicated().sum()
    if date_dupes > 0:
        issues.append(f"❌ {date_dupes} duplicate dates found")
    
    # Check for null values in critical columns
    critical_cols = ['date', 'temperature_2m', 'pm2_5', 'pm10']
    for col in critical_cols:
        if col in df.columns:
            nulls = df[col].isnull().sum()
            if nulls > 0:
                issues.append(f"⚠️  {nulls} null values in {col}")
    
    # Check for unreasonable values
    if 'temperature_2m' in df.columns:
        temp_range = df['temperature_2m'].describe()
        if temp_range['min'] < -50 or temp_range['max'] > 60:
            issues.append(f"⚠️  Temperature values outside reasonable range: {temp_range['min']:.1f}°C to {temp_range['max']:.1f}°C")
    
    if 'pm2_5' in df.columns:
        pm25_max = df['pm2_5'].max()
        if pm25_max > 1000:
            issues.append(f"⚠️  PM2.5 values unusually high: max {pm25_max:.1f}")
    
    # Check data continuity
    df_sorted = df.sort_values('date')
    time_diffs = pd.to_datetime(df_sorted['date']).diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=6)]
    if len(large_gaps) > 0:
        issues.append(f"⚠️  {len(large_gaps)} time gaps larger than 6 hours detected")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ Data integrity validation passed")
        return True

def fetch_data():
    """Main function to fetch and update data"""
    print(f"\n🚀 Starting data fetch at {datetime.now()}")
    print("=" * 50)
    print(f"📁 Data file: {FULL_FILE_PATH}")
    print(f"📍 Location: {LAT}, {LON} (Karachi)")
    
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created data directory: {DATA_DIR}")
    
    try:
        # Get dynamic date range based on existing data
        start_date, end_date = get_date_range(FULL_FILE_PATH)
        
        # Check if we need to fetch data
        if start_date >= end_date:
            print("📊 No new data to fetch (start date >= end date)")
            return
            
        print(f"📅 Fetching data from {start_date} to {end_date}")
        
        # Fetch data
        aqi_df = fetch_aqi_data(start_date, end_date)
        weather_df = fetch_weather_data(start_date, end_date)
        
        # Merge and save
        final_df = merge_and_update(weather_df, aqi_df, FULL_FILE_PATH)
        
        # Validate final data integrity
        if final_df is not None:
            validate_data_integrity(final_df, FULL_FILE_PATH)
        
        print("✅ Data fetch completed successfully")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise

def run_scheduler():
    """Run the scheduler that triggers data fetch every 3 days"""
    print("🔄 Setting up scheduler to run every 3 days...")
    
    # Schedule the job to run every 3 days
    schedule.every(3).days.do(fetch_data)
    
    # Run once immediately
    print("🚀 Running initial data fetch...")
    fetch_data()
    
    print(f"⏰ Next run scheduled in 3 days")
    print("🔄 Scheduler started. Press Ctrl+C to stop.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
    except KeyboardInterrupt:
        print("\n👋 Scheduler stopped by user")

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--schedule":
        # Run with scheduler
        run_scheduler()
    else:
        # Run once
        fetch_data()

if __name__ == "__main__":
    main()
