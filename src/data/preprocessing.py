"""
Data Preprocessing Module

This module handles data cleaning and preprocessing for currency exchange rate data.
Implements single responsibility principle by focusing only on data preprocessing.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from src.data.schema import CurrencyData, Currency, ExchangeRate

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessor for currency exchange rate data"""
    
    def __init__(self):
        """Initialize the data preprocessor"""
        self.processed_data = {}
    
    def convert_to_dataframe(self, currency_data: CurrencyData) -> pd.DataFrame:
        """
        Convert currency data to pandas DataFrame for easier processing
        
        Args:
            currency_data: CurrencyData object
            
        Returns:
            pd.DataFrame: DataFrame with exchange rate data
        """
        data_rows = []
        
        for currency in currency_data.data:
            for rate in currency.exchange_rates:
                data_rows.append({
                    'currency_id': currency.id,
                    'currency_code': currency.code,
                    'currency_name_fr': currency.name_fr,
                    'currency_name_ar': currency.name_ar,
                    'unity': currency.unity,
                    'rate_id': rate.id,
                    'day': pd.to_datetime(rate.day),
                    'value': float(rate.value),
                    'end_date': pd.to_datetime(rate.end_date)
                })
        
        df = pd.DataFrame(data_rows)
        df = df.sort_values(['currency_code', 'day'])
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            method: Method to handle missing values ('interpolate', 'forward_fill', 'backward_fill', 'drop')
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        # Check for missing values
        missing_count = df_clean.isnull().sum()
        if missing_count.sum() > 0:
            logger.info(f"Found missing values: {missing_count.to_dict()}")
        
        if method == 'interpolate':
            # Interpolate missing values for each currency separately
            for currency_code in df_clean['currency_code'].unique():
                currency_mask = df_clean['currency_code'] == currency_code
                df_clean.loc[currency_mask, 'value'] = df_clean.loc[currency_mask, 'value'].interpolate(method='linear')
        
        elif method == 'forward_fill':
            df_clean['value'] = df_clean.groupby('currency_code')['value'].fillna(method='ffill')
        
        elif method == 'backward_fill':
            df_clean['value'] = df_clean.groupby('currency_code')['value'].fillna(method='bfill')
        
        elif method == 'drop':
            df_clean = df_clean.dropna(subset=['value'])
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df_clean
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                                 threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and handle outliers in exchange rate values
        
        Args:
            df: Input DataFrame
            method: Method to detect outliers ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        for currency_code in df_clean['currency_code'].unique():
            currency_mask = df_clean['currency_code'] == currency_code
            currency_data = df_clean.loc[currency_mask, 'value']
            
            if method == 'iqr':
                # Interquartile Range method
                Q1 = currency_data.quantile(0.25)
                Q3 = currency_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (currency_data < lower_bound) | (currency_data > upper_bound)
            
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((currency_data - currency_data.mean()) / currency_data.std())
                outliers = z_scores > threshold
            
            elif method == 'isolation_forest':
                # Isolation Forest method (simplified)
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(currency_data.values.reshape(-1, 1))
                outliers = outlier_labels == -1
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                logger.info(f"Found {outlier_count} outliers in {currency_code}")
                
                # Replace outliers with interpolated values
                df_clean.loc[currency_mask & outliers, 'value'] = np.nan
                df_clean.loc[currency_mask, 'value'] = df_clean.loc[currency_mask, 'value'].interpolate(method='linear')
        
        return df_clean
    
    def normalize_values(self, df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize exchange rate values
        
        Args:
            df: Input DataFrame
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            pd.DataFrame: DataFrame with normalized values
        """
        df_normalized = df.copy()
        
        for currency_code in df_normalized['currency_code'].unique():
            currency_mask = df_normalized['currency_code'] == currency_code
            currency_data = df_normalized.loc[currency_mask, 'value']
            
            if method == 'minmax':
                # Min-Max normalization
                min_val = currency_data.min()
                max_val = currency_data.max()
                df_normalized.loc[currency_mask, 'value_normalized'] = (currency_data - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                # Z-score normalization
                mean_val = currency_data.mean()
                std_val = currency_data.std()
                df_normalized.loc[currency_mask, 'value_normalized'] = (currency_data - mean_val) / std_val
            
            elif method == 'robust':
                # Robust normalization using median and IQR
                median_val = currency_data.median()
                Q1 = currency_data.quantile(0.25)
                Q3 = currency_data.quantile(0.75)
                IQR = Q3 - Q1
                df_normalized.loc[currency_mask, 'value_normalized'] = (currency_data - median_val) / IQR
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return df_normalized
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        df_with_indicators = df.copy()
        
        for currency_code in df_with_indicators['currency_code'].unique():
            currency_mask = df_with_indicators['currency_code'] == currency_code
            currency_data = df_with_indicators.loc[currency_mask].sort_values('day')
            
            # Moving averages
            df_with_indicators.loc[currency_mask, 'sma_7'] = currency_data['value'].rolling(window=7).mean()
            df_with_indicators.loc[currency_mask, 'sma_30'] = currency_data['value'].rolling(window=30).mean()
            df_with_indicators.loc[currency_mask, 'ema_12'] = currency_data['value'].ewm(span=12).mean()
            
            # Volatility indicators
            df_with_indicators.loc[currency_mask, 'volatility_7'] = currency_data['value'].rolling(window=7).std()
            df_with_indicators.loc[currency_mask, 'volatility_30'] = currency_data['value'].rolling(window=30).std()
            
            # Price changes
            df_with_indicators.loc[currency_mask, 'price_change'] = currency_data['value'].diff()
            df_with_indicators.loc[currency_mask, 'price_change_pct'] = currency_data['value'].pct_change()
            
            # Bollinger Bands
            sma_20 = currency_data['value'].rolling(window=20).mean()
            std_20 = currency_data['value'].rolling(window=20).std()
            df_with_indicators.loc[currency_mask, 'bb_upper'] = sma_20 + (2 * std_20)
            df_with_indicators.loc[currency_mask, 'bb_lower'] = sma_20 - (2 * std_20)
            df_with_indicators.loc[currency_mask, 'bb_position'] = (currency_data['value'] - df_with_indicators.loc[currency_mask, 'bb_lower']) / (df_with_indicators.loc[currency_mask, 'bb_upper'] - df_with_indicators.loc[currency_mask, 'bb_lower'])
        
        return df_with_indicators
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with time features
        """
        df_with_time = df.copy()
        
        # Extract time components
        df_with_time['year'] = df_with_time['day'].dt.year
        df_with_time['month'] = df_with_time['day'].dt.month
        df_with_time['day_of_week'] = df_with_time['day'].dt.dayofweek
        df_with_time['day_of_year'] = df_with_time['day'].dt.dayofyear
        df_with_time['quarter'] = df_with_time['day'].dt.quarter
        
        # Cyclical encoding for periodic features
        df_with_time['month_sin'] = np.sin(2 * np.pi * df_with_time['month'] / 12)
        df_with_time['month_cos'] = np.cos(2 * np.pi * df_with_time['month'] / 12)
        df_with_time['day_of_week_sin'] = np.sin(2 * np.pi * df_with_time['day_of_week'] / 7)
        df_with_time['day_of_week_cos'] = np.cos(2 * np.pi * df_with_time['day_of_week'] / 7)
        
        return df_with_time
    
    def resample_data(self, df: pd.DataFrame, frequency: str = 'D', 
                     method: str = 'last') -> pd.DataFrame:
        """
        Resample data to different frequency
        
        Args:
            df: Input DataFrame
            frequency: Target frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
            method: Aggregation method ('last', 'mean', 'median', 'min', 'max')
            
        Returns:
            pd.DataFrame: Resampled DataFrame
        """
        resampled_data = []
        
        for currency_code in df['currency_code'].unique():
            currency_mask = df['currency_code'] == currency_code
            currency_data = df.loc[currency_mask].set_index('day')
            
            # Resample based on method
            if method == 'last':
                resampled = currency_data.resample(frequency).last()
            elif method == 'mean':
                resampled = currency_data.resample(frequency).mean()
            elif method == 'median':
                resampled = currency_data.resample(frequency).median()
            elif method == 'min':
                resampled = currency_data.resample(frequency).min()
            elif method == 'max':
                resampled = currency_data.resample(frequency).max()
            else:
                raise ValueError(f"Unknown method: {method}")
            
            resampled = resampled.reset_index()
            resampled['currency_code'] = currency_code
            resampled_data.append(resampled)
        
        return pd.concat(resampled_data, ignore_index=True)
    
    def create_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """
        Create lagged features for time series analysis
        
        Args:
            df: Input DataFrame
            lags: List of lag periods
            
        Returns:
            pd.DataFrame: DataFrame with lagged features
        """
        df_with_lags = df.copy()
        
        for currency_code in df_with_lags['currency_code'].unique():
            currency_mask = df_with_lags['currency_code'] == currency_code
            currency_data = df_with_lags.loc[currency_mask].sort_values('day')
            
            for lag in lags:
                df_with_lags.loc[currency_mask, f'value_lag_{lag}'] = currency_data['value'].shift(lag)
                df_with_lags.loc[currency_mask, f'price_change_lag_{lag}'] = currency_data['price_change'].shift(lag)
        
        return df_with_lags
    
    def get_preprocessing_summary(self, original_df: pd.DataFrame, 
                                processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate preprocessing summary report
        
        Args:
            original_df: Original DataFrame
            processed_df: Processed DataFrame
            
        Returns:
            Dict: Preprocessing summary
        """
        summary = {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'missing_values_removed': original_df.isnull().sum().sum() - processed_df.isnull().sum().sum(),
            'currencies_processed': processed_df['currency_code'].nunique(),
            'date_range': {
                'start': processed_df['day'].min().strftime('%Y-%m-%d'),
                'end': processed_df['day'].max().strftime('%Y-%m-%d')
            },
            'features_added': list(set(processed_df.columns) - set(original_df.columns))
        }
        
        return summary
    
    def preprocess_complete(self, currency_data: CurrencyData, 
                          config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline
        
        Args:
            currency_data: CurrencyData object
            config: Preprocessing configuration
            
        Returns:
            Tuple: (processed DataFrame, preprocessing summary)
        """
        if config is None:
            config = {
                'handle_missing': 'interpolate',
                'handle_outliers': 'iqr',
                'outlier_threshold': 1.5,
                'normalize': 'minmax',
                'add_indicators': True,
                'add_time_features': True,
                'resample': None,
                'lags': [1, 7, 30]
            }
        
        logger.info("Starting complete preprocessing pipeline")
        
        # Convert to DataFrame
        df = self.convert_to_dataframe(currency_data)
        original_shape = df.shape
        
        # Handle missing values
        if config.get('handle_missing'):
            df = self.handle_missing_values(df, method=config['handle_missing'])
        
        # Handle outliers
        if config.get('handle_outliers'):
            df = self.detect_and_handle_outliers(df, method=config['handle_outliers'], 
                                               threshold=config.get('outlier_threshold', 1.5))
        
        # Normalize values
        if config.get('normalize'):
            df = self.normalize_values(df, method=config['normalize'])
        
        # Add technical indicators
        if config.get('add_indicators', True):
            df = self.add_technical_indicators(df)
        
        # Add time features
        if config.get('add_time_features', True):
            df = self.add_time_features(df)
        
        # Resample if specified
        if config.get('resample'):
            df = self.resample_data(df, frequency=config['resample']['frequency'], 
                                  method=config['resample']['method'])
        
        # Add lagged features
        if config.get('lags'):
            df = self.create_lagged_features(df, lags=config['lags'])
        
        # Generate summary
        summary = self.get_preprocessing_summary(pd.DataFrame(original_shape), df)
        
        logger.info(f"Preprocessing completed. Shape: {original_shape} -> {df.shape}")
        
        return df, summary
