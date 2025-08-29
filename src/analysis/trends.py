"""
Historical Trend Analysis Module

This module handles trend calculation and analysis for currency exchange rate data.
Implements single responsibility principle by focusing only on trend analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Analyzer for currency exchange rate trends"""
    
    def __init__(self):
        """Initialize the trend analyzer"""
        self.trend_results = {}
    
    def calculate_linear_trend(self, df: pd.DataFrame, currency_code: str = None) -> Dict[str, Any]:
        """
        Calculate linear trend for currency data
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze (if None, analyze all)
            
        Returns:
            Dict: Trend analysis results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        # Handle empty or insufficient data
        if len(df_filtered) == 0:
            return {
                'currency_code': currency_code or 'all',
                'slope': 0.0,
                'intercept': 0.0,
                'r2_score': 0.0,
                'rmse': 0.0,
                'trend_direction': 'stable',
                'total_change_percent': 0.0,
                'volatility': 0.0,
                'data_points': 0,
                'start_value': 0.0,
                'end_value': 0.0,
                'predictions': [],
                'actual_values': [],
                'dates': []
            }
        
        if len(df_filtered) == 1:
            # Single data point - no trend
            value = df_filtered['value'].iloc[0]
            return {
                'currency_code': currency_code or 'all',
                'slope': 0.0,
                'intercept': value,
                'r2_score': 0.0,
                'rmse': 0.0,
                'trend_direction': 'stable',
                'total_change_percent': 0.0,
                'volatility': 0.0,
                'data_points': 1,
                'start_value': value,
                'end_value': value,
                'predictions': [value],
                'actual_values': [value],
                'dates': [df_filtered['day'].iloc[0].strftime('%Y-%m-%d')]
            }
        
        # Create time index
        df_filtered['time_index'] = range(len(df_filtered))
        
        # Linear regression
        X = df_filtered['time_index'].values.reshape(-1, 1)
        y = df_filtered['value'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate trend direction and strength
        slope = model.coef_[0]
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        
        # Calculate percentage change
        total_change = ((y[-1] - y[0]) / y[0]) * 100 if y[0] != 0 else 0
        
        # Calculate volatility
        volatility = np.std(y)
        
        result = {
            'currency_code': currency_code or 'all',
            'slope': slope,
            'intercept': model.intercept_,
            'r2_score': r2,
            'rmse': rmse,
            'trend_direction': trend_direction,
            'total_change_percent': total_change,
            'volatility': volatility,
            'data_points': len(y),
            'start_value': y[0],
            'end_value': y[-1],
            'predictions': y_pred.tolist(),
            'actual_values': y.tolist(),
            'dates': df_filtered['day'].dt.strftime('%Y-%m-%d').tolist()
        }
        
        return result
    
    def calculate_polynomial_trend(self, df: pd.DataFrame, degree: int = 2, 
                                 currency_code: str = None) -> Dict[str, Any]:
        """
        Calculate polynomial trend for currency data
        
        Args:
            df: DataFrame with currency data
            degree: Polynomial degree
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Polynomial trend analysis results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        # Create time index
        df_filtered['time_index'] = range(len(df_filtered))
        
        # Polynomial regression
        X = df_filtered['time_index'].values.reshape(-1, 1)
        y = df_filtered['value'].values
        
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Calculate predictions
        y_pred = model.predict(X_poly)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate trend direction based on overall change
        total_change = ((y[-1] - y[0]) / y[0]) * 100 if y[0] != 0 else 0
        trend_direction = "increasing" if total_change > 0 else "decreasing" if total_change < 0 else "stable"
        
        result = {
            'currency_code': currency_code or 'all',
            'degree': degree,
            'coefficients': model.coef_.tolist(),
            'intercept': model.intercept_,
            'r2_score': r2,
            'rmse': rmse,
            'trend_direction': trend_direction,
            'total_change_percent': total_change,
            'data_points': len(y),
            'predictions': y_pred.tolist(),
            'actual_values': y.tolist(),
            'dates': df_filtered['day'].dt.strftime('%Y-%m-%d').tolist()
        }
        
        return result
    
    def calculate_moving_averages(self, df: pd.DataFrame, windows: List[int] = [7, 14, 30],
                                currency_code: str = None) -> Dict[str, Any]:
        """
        Calculate moving averages for currency data
        
        Args:
            df: DataFrame with currency data
            windows: List of window sizes for moving averages
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Moving average results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        result = {
            'currency_code': currency_code or 'all',
            'dates': df_filtered['day'].dt.strftime('%Y-%m-%d').tolist(),
            'actual_values': df_filtered['value'].tolist(),
            'moving_averages': {}
        }
        
        for window in windows:
            if len(df_filtered) >= window:
                ma = df_filtered['value'].rolling(window=window).mean()
                result['moving_averages'][f'ma_{window}'] = ma.tolist()
            else:
                result['moving_averages'][f'ma_{window}'] = [None] * len(df_filtered)
        
        return result
    
    def calculate_exponential_moving_average(self, df: pd.DataFrame, spans: List[float] = [12, 26],
                                           currency_code: str = None) -> Dict[str, Any]:
        """
        Calculate exponential moving averages for currency data
        
        Args:
            df: DataFrame with currency data
            spans: List of span values for EMA
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Exponential moving average results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        result = {
            'currency_code': currency_code or 'all',
            'dates': df_filtered['day'].dt.strftime('%Y-%m-%d').tolist(),
            'actual_values': df_filtered['value'].tolist(),
            'exponential_moving_averages': {}
        }
        
        for span in spans:
            ema = df_filtered['value'].ewm(span=span).mean()
            result['exponential_moving_averages'][f'ema_{span}'] = ema.tolist()
        
        return result
    
    def detect_trend_changes(self, df: pd.DataFrame, window: int = 10,
                           currency_code: str = None) -> Dict[str, Any]:
        """
        Detect trend changes using moving average crossovers
        
        Args:
            df: DataFrame with currency data
            window: Window size for trend detection
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Trend change detection results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < window * 2:
            return {
                'currency_code': currency_code or 'all',
                'trend_changes': [],
                'total_changes': 0,
                'bullish_changes': 0,
                'bearish_changes': 0,
                'message': 'Insufficient data for trend change detection'
            }
        
        # Calculate short and long moving averages
        short_ma = df_filtered['value'].rolling(window=window).mean()
        long_ma = df_filtered['value'].rolling(window=window*2).mean()
        
        # Detect crossovers
        crossover_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        crossover_down = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        
        trend_changes = []
        
        for i in range(1, len(df_filtered)):
            if crossover_up.iloc[i]:
                trend_changes.append({
                    'date': df_filtered['day'].iloc[i].strftime('%Y-%m-%d'),
                    'type': 'bullish',
                    'value': df_filtered['value'].iloc[i],
                    'short_ma': short_ma.iloc[i],
                    'long_ma': long_ma.iloc[i]
                })
            elif crossover_down.iloc[i]:
                trend_changes.append({
                    'date': df_filtered['day'].iloc[i].strftime('%Y-%m-%d'),
                    'type': 'bearish',
                    'value': df_filtered['value'].iloc[i],
                    'short_ma': short_ma.iloc[i],
                    'long_ma': long_ma.iloc[i]
                })
        
        result = {
            'currency_code': currency_code or 'all',
            'trend_changes': trend_changes,
            'total_changes': len(trend_changes),
            'bullish_changes': len([c for c in trend_changes if c['type'] == 'bullish']),
            'bearish_changes': len([c for c in trend_changes if c['type'] == 'bearish'])
        }
        
        return result
        
        return result
    
    def calculate_trend_strength(self, df: pd.DataFrame, currency_code: str = None) -> Dict[str, Any]:
        """
        Calculate trend strength using various indicators
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Trend strength analysis results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        # Calculate price changes
        df_filtered['price_change'] = df_filtered['value'].diff()
        df_filtered['price_change_pct'] = df_filtered['value'].pct_change()
        
        # Calculate trend strength indicators
        positive_days = (df_filtered['price_change'] > 0).sum()
        negative_days = (df_filtered['price_change'] < 0).sum()
        total_days = len(df_filtered) - 1
        
        # Calculate average daily change
        avg_daily_change = df_filtered['price_change'].mean()
        avg_daily_change_pct = df_filtered['price_change_pct'].mean()
        
        # Calculate consistency (how often the trend continues)
        trend_consistency = positive_days / total_days if total_days > 0 else 0
        
        # Calculate momentum (rate of change)
        momentum = df_filtered['price_change_pct'].rolling(window=5).mean().iloc[-1] if len(df_filtered) >= 5 else 0
        
        # Determine trend strength category
        if abs(trend_consistency - 0.5) < 0.1:
            strength_category = "weak"
        elif abs(trend_consistency - 0.5) < 0.2:
            strength_category = "moderate"
        else:
            strength_category = "strong"
        
        result = {
            'currency_code': currency_code or 'all',
            'positive_days': positive_days,
            'negative_days': negative_days,
            'total_days': total_days,
            'trend_consistency': trend_consistency,
            'avg_daily_change': avg_daily_change,
            'avg_daily_change_pct': avg_daily_change_pct,
            'momentum': momentum,
            'strength_category': strength_category,
            'trend_direction': "bullish" if trend_consistency > 0.5 else "bearish" if trend_consistency < 0.5 else "neutral"
        }
        
        return result
    
    def analyze_all_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive trend analysis for all currencies
        
        Args:
            df: DataFrame with currency data
            
        Returns:
            Dict: Comprehensive trend analysis results
        """
        currencies = df['currency_code'].unique()
        
        results = {
            'currencies': {},
            'summary': {
                'total_currencies': len(currencies),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        for currency in currencies:
            currency_results = {
                'linear_trend': self.calculate_linear_trend(df, currency),
                'polynomial_trend': self.calculate_polynomial_trend(df, degree=2, currency_code=currency),
                'moving_averages': self.calculate_moving_averages(df, currency_code=currency),
                'ema': self.calculate_exponential_moving_average(df, currency_code=currency),
                'trend_changes': self.detect_trend_changes(df, currency_code=currency),
                'trend_strength': self.calculate_trend_strength(df, currency_code=currency)
            }
            
            results['currencies'][currency] = currency_results
        
        # Calculate overall market trends
        overall_linear = self.calculate_linear_trend(df)
        results['overall_market'] = {
            'linear_trend': overall_linear,
            'trend_strength': self.calculate_trend_strength(df)
        }
        
        return results
