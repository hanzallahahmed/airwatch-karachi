import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class OptimizedAQIFeatureEngineering:
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.scaler = None
        self.feature_names = []
        
        # Define feature groups
        self.aqi_features = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
                            "sulphur_dioxide", "ozone"]
        self.weather_features = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", 
                               "surface_pressure", "wind_speed_10m", "wind_direction_10m", 
                               "precipitation", "cloud_cover"]
        self.all_features = self.aqi_features + self.weather_features
    
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading data...")
        self.data = pd.read_parquet(self.filename)
        
        print(f"Initial data shape: {self.data.shape}")
        
        # Handle datetime index/column
        if 'date' in self.data.columns:
            self.data.set_index('date', inplace=True)
        elif not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except:
                start_date = '2023-01-01'
                self.data.index = pd.date_range(start=start_date, periods=len(self.data), freq='H')
        
        # Sort and remove duplicates
        self.data = self.data.sort_index()
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        
        print(f"Final data shape: {self.data.shape}")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        
        return self.data
    
    def create_temporal_features(self):
        """Create essential temporal features only"""
        print("Creating temporal features...")
        
        # Basic temporal features
        self.data['hour'] = self.data.index.hour
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['is_weekend'] = (self.data.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding for hour and month only (most important)
        self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)
        
        # Rush hour indicators
        self.data['rush_hour'] = (
            ((self.data['hour'] >= 7) & (self.data['hour'] <= 9)) |
            ((self.data['hour'] >= 17) & (self.data['hour'] <= 19))
        ).astype(int)
        
        # Day/night
        self.data['is_day'] = ((self.data['hour'] >= 6) & (self.data['hour'] <= 18)).astype(int)
    
    def create_lag_features(self, lag_hours=[1, 6, 24]):
        """Create only essential lag features"""
        print("Creating lag features...")
        
        # Only create lag features for most important pollutants and weather
        important_features = ['pm2_5', 'pm10', 'temperature_2m', 'wind_speed_10m', 'relative_humidity_2m']
        
        for feature in important_features:
            if feature in self.data.columns:
                for lag in lag_hours:
                    self.data[f"{feature}_lag_{lag}h"] = self.data[feature].shift(lag)
    
    def create_rolling_features(self, windows=[3, 12, 24]):
        """Create essential rolling features with vectorized operations"""
        print("Creating rolling features...")
        
        # Only for most important features
        important_features = ['pm2_5', 'pm10', 'temperature_2m', 'wind_speed_10m']
        
        for feature in important_features:
            if feature in self.data.columns:
                for window in windows:
                    # Use vectorized operations
                    rolling = self.data[feature].rolling(window=window, min_periods=1)
                    self.data[f"{feature}_mean_{window}h"] = rolling.mean()
                    self.data[f"{feature}_std_{window}h"] = rolling.std()
                    
                    # Only max/min for PM features
                    if 'pm' in feature:
                        self.data[f"{feature}_max_{window}h"] = rolling.max()
    
    def create_interaction_features(self):
        """Create key interaction features"""
        print("Creating interaction features...")
        
        # Wind-pollutant interactions (most important)
        for pollutant in ['pm2_5', 'pm10']:
            if pollutant in self.data.columns and 'wind_speed_10m' in self.data.columns:
                wind_speed_safe = self.data['wind_speed_10m'].clip(lower=0.1)
                self.data[f"{pollutant}_wind_ratio"] = self.data[pollutant] / wind_speed_safe
        
        # Temperature-humidity interaction
        if 'temperature_2m' in self.data.columns and 'relative_humidity_2m' in self.data.columns:
            self.data['temp_humidity_index'] = self.data['temperature_2m'] * (self.data['relative_humidity_2m'] / 100)
    
    def create_atmospheric_features(self):
        """Create key atmospheric features"""
        print("Creating atmospheric features...")
        
        # Atmospheric stability
        if 'temperature_2m' in self.data.columns and 'wind_speed_10m' in self.data.columns:
            wind_safe = self.data['wind_speed_10m'].clip(lower=0.1)
            self.data['atmospheric_stability'] = self.data['temperature_2m'] / wind_safe
        
        # Vapor pressure deficit
        if 'temperature_2m' in self.data.columns and 'dew_point_2m' in self.data.columns:
            self.data['vapor_pressure_deficit'] = self.data['temperature_2m'] - self.data['dew_point_2m']
    
    def create_pollution_indices(self):
        """Create composite pollution indices"""
        print("Creating pollution indices...")
        
        pollutants = [col for col in self.aqi_features if col in self.data.columns]
        
        if len(pollutants) > 1:
            # Simple mean for composite index
            self.data['pollution_index'] = self.data[pollutants].mean(axis=1)
            
            # PM ratio if both PM2.5 and PM10 exist
            if 'pm2_5' in self.data.columns and 'pm10' in self.data.columns:
                pm10_safe = self.data['pm10'].clip(lower=0.1)
                self.data['pm_ratio'] = self.data['pm2_5'] / pm10_safe
    
    def create_seasonal_patterns(self):
        """Create seasonal indicators"""
        print("Creating seasonal patterns...")
        
        # Season indicators
        self.data['season'] = self.data['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3  # Autumn
        })
        
        # Weather pattern indicators
        if 'precipitation' in self.data.columns:
            self.data['has_rain'] = (self.data['precipitation'] > 0).astype(int)
        
        if 'wind_speed_10m' in self.data.columns:
            self.data['calm_wind'] = (self.data['wind_speed_10m'] < 2).astype(int)
    
    def handle_missing_values(self, method='forward_fill'):
        """Optimized missing value handling"""
        print("Handling missing values...")
        
        print(f"Missing values before: {self.data.isnull().sum().sum()}")
        
        if method == 'forward_fill':
            # Forward fill + backward fill (fastest)
            self.data = self.data.fillna(method='ffill', limit=6)
            self.data = self.data.fillna(method='bfill', limit=6)
            # Interpolate remaining
            self.data = self.data.interpolate(method='linear', limit=12)
            
        elif method == 'knn':
            # KNN only for numeric columns with few missing values
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            missing_pct = self.data[numeric_cols].isnull().sum() / len(self.data)
            cols_to_impute = missing_pct[missing_pct < 0.3].index  # Only < 30% missing
            
            if len(cols_to_impute) > 0:
                imputer = KNNImputer(n_neighbors=5)
                self.data[cols_to_impute] = imputer.fit_transform(self.data[cols_to_impute])
        
        # Fill any remaining with median
        for col in self.data.select_dtypes(include=[np.number]).columns:
            if self.data[col].isnull().any():
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        print(f"Missing values after: {self.data.isnull().sum().sum()}")
    
    def normalize_features(self, method='standard'):
        """Normalize features efficiently"""
        print("Normalizing features...")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
    
    def feature_selection(self, target_col, method='correlation', k=50):
        """Fast feature selection"""
        print(f"Performing feature selection...")
        
        if target_col not in self.data.columns:
            print(f"Target column {target_col} not found!")
            return []
        
        # Remove target from features
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        
        # Remove rows with missing target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if method == 'correlation':
            # Fast correlation-based selection
            correlations = pd.concat([X, y], axis=1).corr()[target_col].abs().sort_values(ascending=False)
            selected_features = correlations.head(k+1).index.tolist()
            if target_col in selected_features:
                selected_features.remove(target_col)
            
        elif method == 'mutual_info':
            # Sample data if too large
            if len(X) > 5000:
                sample_idx = np.random.choice(len(X), 5000, replace=False)
                X_sample = X.iloc[sample_idx]
                y_sample = y.iloc[sample_idx]
            else:
                X_sample, y_sample = X, y
            
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
            selector.fit(X_sample, y_sample)
            selected_features = X.columns[selector.get_support()].tolist()
        
        # Keep only selected features plus target
        self.data = self.data[selected_features + [target_col]]
        
        print(f"Selected {len(selected_features)} features")
        return selected_features
    
    def run_feature_engineering(self, target_col=None, normalize=True, 
                              imputation_method='forward_fill', 
                              feature_selection_method='correlation', 
                              max_features=50):
        """Run optimized feature engineering pipeline"""
        print("Starting optimized feature engineering pipeline...")
        
        # Load data
        self.load_data()
        
        # Create features efficiently
        self.create_temporal_features()
        self.create_lag_features()
        self.create_rolling_features()
        self.create_interaction_features()
        self.create_atmospheric_features()
        self.create_pollution_indices()
        self.create_seasonal_patterns()
        
        print(f"Features created. Shape: {self.data.shape}")
        
        # Drop initial rows affected by lag operations
        self.data = self.data.iloc[24:]  # Only drop 24 hours instead of 72
        print(f"Shape after dropping lag-affected rows: {self.data.shape}")
        
        # Handle missing values
        self.handle_missing_values(method=imputation_method)
        
        # Feature selection before normalization (more efficient)
        if feature_selection_method and target_col:
            selected_features = self.feature_selection(target_col, feature_selection_method, max_features)
        
        # Normalize features
        if normalize:
            self.normalize_features()
        
        print(f"Feature engineering complete! Final shape: {self.data.shape}")
        
        return self.data
    
    def save_engineered_data(self, output_filename):
        """Save the engineered dataset"""
        self.data.to_parquet(output_filename)
        print(f"Engineered data saved to {output_filename}")

# Example usage with optimized settings
if __name__ == "__main__":
    # Initialize feature engineering
    fe = OptimizedAQIFeatureEngineering("karachi_aqi_data.parquet")
    
    # Run optimized pipeline
    engineered_data = fe.run_feature_engineering(
        target_col='pm2_5',
        normalize=True,
        imputation_method='forward_fill',  # Much faster than iterative
        feature_selection_method='correlation',  # Faster than mutual_info
        max_features=50  # Reasonable number of features
    )
    
    # Save engineered data
    fe.save_engineered_data("karachi_aqi_engineered_optimized.parquet")
    
    # Display summary
    print("\n=== OPTIMIZED FEATURE ENGINEERING SUMMARY ===")
    print(f"Final features: {len(engineered_data.columns)}")
    print(f"Dataset shape: {engineered_data.shape}")
    print(f"Date range: {engineered_data.index.min()} to {engineered_data.index.max()}")
