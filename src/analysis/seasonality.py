"""
Seasonal Pattern Detection Module

This module handles seasonal pattern detection and analysis for currency exchange rate data.
Implements single responsibility principle by focusing only on seasonality analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SeasonalityAnalyzer:
    """Analyzer for currency exchange rate seasonality"""
    
    def __init__(self):
        """Initialize the seasonality analyzer"""
        self.seasonality_results = {}
    
    def detect_seasonal_patterns(self, df: pd.DataFrame, currency_code: str = None,
                               period: int = 12) -> Dict[str, Any]:
        """
        Detect seasonal patterns in currency data
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            period: Seasonal period (e.g., 12 for monthly, 4 for quarterly)
            
        Returns:
            Dict: Seasonal pattern detection results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < period * 2:
            return {
                'currency_code': currency_code or 'all',
                'period': period,
                'message': f'Insufficient data for seasonal analysis (minimum {period * 2} observations required)'
            }
        
        # Prepare time series data
        df_filtered = df_filtered.set_index('day')
        ts_data = df_filtered['value']
        
        # Perform seasonal decomposition
        try:
            decomposition = seasonal_decompose(ts_data, period=period, extrapolate_trend='freq')
            
            # Extract components
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            # Calculate seasonal strength
            seasonal_strength = self._calculate_seasonal_strength(ts_data, seasonal, residual)
            
            # Detect seasonal peaks and troughs
            seasonal_peaks = self._find_seasonal_peaks(seasonal, period)
            
            result = {
                'currency_code': currency_code or 'all',
                'period': period,
                'seasonal_strength': seasonal_strength,
                'trend': trend.tolist() if trend is not None else [],
                'seasonal': seasonal.tolist() if seasonal is not None else [],
                'residual': residual.tolist() if residual is not None else [],
                'seasonal_peaks': seasonal_peaks,
                'data_points': len(ts_data),
                'date_range': {
                    'start': ts_data.index.min().strftime('%Y-%m-%d'),
                    'end': ts_data.index.max().strftime('%Y-%m-%d')
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'currency_code': currency_code or 'all',
                'period': period,
                'message': f'Error in seasonal decomposition: {str(e)}'
            }
    
    def analyze_monthly_patterns(self, df: pd.DataFrame, currency_code: str = None) -> Dict[str, Any]:
        """
        Analyze monthly seasonal patterns
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Monthly pattern analysis results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < 24:  # At least 2 years of data
            return {
                'currency_code': currency_code or 'all',
                'message': 'Insufficient data for monthly pattern analysis (minimum 24 months required)'
            }
        
        # Add month information
        df_filtered['month'] = df_filtered['day'].dt.month
        df_filtered['year'] = df_filtered['day'].dt.year
        
        # Calculate monthly averages
        monthly_stats = df_filtered.groupby('month').agg({
            'value': ['mean', 'std', 'min', 'max', 'count']
        }).round(4)
        
        # Flatten column names
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
        monthly_stats = monthly_stats.reset_index()
        
        # Calculate seasonal indices
        overall_mean = df_filtered['value'].mean()
        monthly_stats['seasonal_index'] = (monthly_stats['value_mean'] / overall_mean * 100).round(2)
        
        # Find strongest and weakest months
        strongest_month = monthly_stats.loc[monthly_stats['seasonal_index'].idxmax()]
        weakest_month = monthly_stats.loc[monthly_stats['seasonal_index'].idxmin()]
        
        # Calculate seasonal variation
        seasonal_variation = monthly_stats['seasonal_index'].std()
        
        result = {
            'currency_code': currency_code or 'all',
            'monthly_stats': monthly_stats.to_dict('records'),
            'strongest_month': {
                'month': int(strongest_month['month']),
                'seasonal_index': strongest_month['seasonal_index'],
                'average_value': strongest_month['value_mean']
            },
            'weakest_month': {
                'month': int(weakest_month['month']),
                'seasonal_index': weakest_month['seasonal_index'],
                'average_value': weakest_month['value_mean']
            },
            'seasonal_variation': seasonal_variation,
            'overall_mean': overall_mean,
            'data_points': len(df_filtered)
        }
        
        return result
    
    def analyze_quarterly_patterns(self, df: pd.DataFrame, currency_code: str = None) -> Dict[str, Any]:
        """
        Analyze quarterly seasonal patterns
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Quarterly pattern analysis results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < 8:  # At least 2 years of data
            return {
                'currency_code': currency_code or 'all',
                'message': 'Insufficient data for quarterly pattern analysis (minimum 8 quarters required)'
            }
        
        # Add quarter information
        df_filtered['quarter'] = df_filtered['day'].dt.quarter
        df_filtered['year'] = df_filtered['day'].dt.year
        
        # Calculate quarterly averages
        quarterly_stats = df_filtered.groupby('quarter').agg({
            'value': ['mean', 'std', 'min', 'max', 'count']
        }).round(4)
        
        # Flatten column names
        quarterly_stats.columns = ['_'.join(col).strip() for col in quarterly_stats.columns]
        quarterly_stats = quarterly_stats.reset_index()
        
        # Calculate seasonal indices
        overall_mean = df_filtered['value'].mean()
        quarterly_stats['seasonal_index'] = (quarterly_stats['value_mean'] / overall_mean * 100).round(2)
        
        # Find strongest and weakest quarters
        strongest_quarter = quarterly_stats.loc[quarterly_stats['seasonal_index'].idxmax()]
        weakest_quarter = quarterly_stats.loc[quarterly_stats['seasonal_index'].idxmin()]
        
        # Calculate seasonal variation
        seasonal_variation = quarterly_stats['seasonal_index'].std()
        
        result = {
            'currency_code': currency_code or 'all',
            'quarterly_stats': quarterly_stats.to_dict('records'),
            'strongest_quarter': {
                'quarter': int(strongest_quarter['quarter']),
                'seasonal_index': strongest_quarter['seasonal_index'],
                'average_value': strongest_quarter['value_mean']
            },
            'weakest_quarter': {
                'quarter': int(weakest_quarter['quarter']),
                'seasonal_index': weakest_quarter['seasonal_index'],
                'average_value': weakest_quarter['value_mean']
            },
            'seasonal_variation': seasonal_variation,
            'overall_mean': overall_mean,
            'data_points': len(df_filtered)
        }
        
        return result
    
    def analyze_weekly_patterns(self, df: pd.DataFrame, currency_code: str = None) -> Dict[str, Any]:
        """
        Analyze weekly seasonal patterns
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Weekly pattern analysis results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < 14:  # At least 2 weeks of data
            return {
                'currency_code': currency_code or 'all',
                'message': 'Insufficient data for weekly pattern analysis (minimum 14 days required)'
            }
        
        # Add day of week information
        df_filtered['day_of_week'] = df_filtered['day'].dt.dayofweek
        df_filtered['day_name'] = df_filtered['day'].dt.day_name()
        
        # Calculate daily averages
        daily_stats = df_filtered.groupby(['day_of_week', 'day_name']).agg({
            'value': ['mean', 'std', 'min', 'max', 'count']
        }).round(4)
        
        # Flatten column names
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns]
        daily_stats = daily_stats.reset_index()
        
        # Calculate seasonal indices
        overall_mean = df_filtered['value'].mean()
        daily_stats['seasonal_index'] = (daily_stats['value_mean'] / overall_mean * 100).round(2)
        
        # Find strongest and weakest days
        strongest_day = daily_stats.loc[daily_stats['seasonal_index'].idxmax()]
        weakest_day = daily_stats.loc[daily_stats['seasonal_index'].idxmin()]
        
        # Calculate seasonal variation
        seasonal_variation = daily_stats['seasonal_index'].std()
        
        result = {
            'currency_code': currency_code or 'all',
            'daily_stats': daily_stats.to_dict('records'),
            'strongest_day': {
                'day_of_week': int(strongest_day['day_of_week']),
                'day_name': strongest_day['day_name'],
                'seasonal_index': strongest_day['seasonal_index'],
                'average_value': strongest_day['value_mean']
            },
            'weakest_day': {
                'day_of_week': int(weakest_day['day_of_week']),
                'day_name': weakest_day['day_name'],
                'seasonal_index': weakest_day['seasonal_index'],
                'average_value': weakest_day['value_mean']
            },
            'seasonal_variation': seasonal_variation,
            'overall_mean': overall_mean,
            'data_points': len(df_filtered)
        }
        
        return result
    
    def detect_periodicity(self, df: pd.DataFrame, currency_code: str = None) -> Dict[str, Any]:
        """
        Detect periodicity in currency data using spectral analysis
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Periodicity detection results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < 10:
            return {
                'currency_code': currency_code or 'all',
                'message': 'Insufficient data for periodicity detection (minimum 10 observations required)'
            }
        
        # Prepare time series data
        ts_data = df_filtered['value'].values
        
        # Remove trend and mean
        ts_detrended = ts_data - np.mean(ts_data)
        
        # Calculate periodogram
        frequencies, power = periodogram(ts_detrended, fs=1.0)
        
        # Find dominant frequencies
        dominant_freq_idx = np.argsort(power)[-3:]  # Top 3 frequencies
        dominant_frequencies = frequencies[dominant_freq_idx]
        dominant_powers = power[dominant_freq_idx]
        
        # Convert frequencies to periods
        periods = 1.0 / dominant_frequencies
        
        # Calculate significance threshold (95% confidence)
        significance_threshold = np.percentile(power, 95)
        
        # Find significant peaks
        significant_peaks = []
        for i, (freq, pwr, period) in enumerate(zip(dominant_frequencies, dominant_powers, periods)):
            if pwr > significance_threshold:
                significant_peaks.append({
                    'rank': i + 1,
                    'frequency': freq,
                    'power': pwr,
                    'period': period,
                    'significance': 'significant' if pwr > significance_threshold else 'not significant'
                })
        
        result = {
            'currency_code': currency_code or 'all',
            'dominant_frequencies': dominant_frequencies.tolist(),
            'dominant_powers': dominant_powers.tolist(),
            'periods': periods.tolist(),
            'significant_peaks': significant_peaks,
            'significance_threshold': significance_threshold,
            'data_points': len(ts_data)
        }
        
        return result
    
    def test_stationarity(self, df: pd.DataFrame, currency_code: str = None) -> Dict[str, Any]:
        """
        Test for stationarity in currency data
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Stationarity test results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < 10:
            return {
                'currency_code': currency_code or 'all',
                'message': 'Insufficient data for stationarity test (minimum 10 observations required)'
            }
        
        # Prepare time series data
        ts_data = df_filtered['value'].values
        
        # Perform Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(ts_data)
            
            result = {
                'currency_code': currency_code or 'all',
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05,
                'significance_level': 0.05,
                'data_points': len(ts_data)
            }
            
            return result
            
        except Exception as e:
            return {
                'currency_code': currency_code or 'all',
                'message': f'Error in stationarity test: {str(e)}'
            }
    
    def _calculate_seasonal_strength(self, ts_data: pd.Series, seasonal: pd.Series, residual: pd.Series) -> float:
        """
        Calculate seasonal strength using variance ratio
        
        Args:
            ts_data: Original time series
            seasonal: Seasonal component
            residual: Residual component
            
        Returns:
            float: Seasonal strength measure
        """
        if seasonal is None or residual is None:
            return 0.0
        
        # Remove NaN values
        seasonal_clean = seasonal.dropna()
        residual_clean = residual.dropna()
        
        if len(seasonal_clean) == 0 or len(residual_clean) == 0:
            return 0.0
        
        # Calculate variances
        seasonal_var = np.var(seasonal_clean)
        residual_var = np.var(residual_clean)
        
        if residual_var == 0:
            return 1.0 if seasonal_var > 0 else 0.0
        
        # Seasonal strength = seasonal variance / (seasonal variance + residual variance)
        seasonal_strength = seasonal_var / (seasonal_var + residual_var)
        
        return min(seasonal_strength, 1.0)  # Cap at 1.0
    
    def _find_seasonal_peaks(self, seasonal: pd.Series, period: int) -> List[Dict[str, Any]]:
        """
        Find seasonal peaks and troughs
        
        Args:
            seasonal: Seasonal component
            period: Seasonal period
            
        Returns:
            List: Seasonal peaks and troughs
        """
        if seasonal is None or len(seasonal) < period:
            return []
        
        seasonal_clean = seasonal.dropna()
        if len(seasonal_clean) == 0:
            return []
        
        peaks = []
        
        # Find peaks within each period
        for i in range(period):
            period_indices = range(i, len(seasonal_clean), period)
            period_values = [seasonal_clean.iloc[j] for j in period_indices if j < len(seasonal_clean)]
            
            if len(period_values) > 0:
                max_val = max(period_values)
                min_val = min(period_values)
                
                peaks.append({
                    'period_position': i,
                    'max_value': max_val,
                    'min_value': min_val,
                    'amplitude': max_val - min_val
                })
        
        return peaks
    
    def analyze_all_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive seasonality analysis for all currencies
        
        Args:
            df: DataFrame with currency data
            
        Returns:
            Dict: Comprehensive seasonality analysis results
        """
        currencies = df['currency_code'].unique()
        
        results = {
            'summary': {
                'total_currencies': len(currencies),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        for currency in currencies:
            currency_results = {
                'seasonal_patterns': self.detect_seasonal_patterns(df, currency_code=currency),
                'monthly_patterns': self.analyze_monthly_patterns(df, currency_code=currency),
                'quarterly_patterns': self.analyze_quarterly_patterns(df, currency_code=currency),
                'weekly_patterns': self.analyze_weekly_patterns(df, currency_code=currency),
                'periodicity': self.detect_periodicity(df, currency_code=currency),
                'stationarity': self.test_stationarity(df, currency_code=currency)
            }
            
            results[currency] = currency_results
        
        # Calculate overall market seasonality
        overall_seasonal = self.detect_seasonal_patterns(df)
        results['overall_market'] = {
            'seasonal_patterns': overall_seasonal
        }
        
        return results
