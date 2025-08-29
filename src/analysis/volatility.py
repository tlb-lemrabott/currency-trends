"""
Volatility Analysis Module

This module handles volatility calculation and analysis for currency exchange rate data.
Implements single responsibility principle by focusing only on volatility analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class VolatilityAnalyzer:
    """Analyzer for currency exchange rate volatility"""
    
    def __init__(self):
        """Initialize the volatility analyzer"""
        self.volatility_results = {}
    
    def calculate_rolling_volatility(self, df: pd.DataFrame, window: int = 20,
                                   currency_code: str = None) -> Dict[str, Any]:
        """
        Calculate rolling volatility for currency data
        
        Args:
            df: DataFrame with currency data
            window: Rolling window size for volatility calculation
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Rolling volatility results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < 2:
            return {
                'currency_code': currency_code or 'all',
                'volatility': [],
                'dates': [],
                'window': window,
                'message': 'Insufficient data for volatility calculation'
            }
        
        # Calculate returns
        df_filtered['returns'] = df_filtered['value'].pct_change()
        
        # Calculate rolling volatility (standard deviation of returns)
        rolling_vol = df_filtered['returns'].rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        result = {
            'currency_code': currency_code or 'all',
            'volatility': rolling_vol.tolist(),
            'returns': df_filtered['returns'].tolist(),
            'dates': df_filtered['day'].dt.strftime('%Y-%m-%d').tolist(),
            'window': window,
            'annualized': True
        }
        
        return result
    
    def calculate_garch_volatility(self, df: pd.DataFrame, currency_code: str = None) -> Dict[str, Any]:
        """
        Calculate GARCH-based volatility estimation
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: GARCH volatility results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < 30:
            return {
                'currency_code': currency_code or 'all',
                'garch_volatility': [],
                'dates': [],
                'message': 'Insufficient data for GARCH estimation (minimum 30 observations)'
            }
        
        # Calculate returns
        df_filtered['returns'] = df_filtered['value'].pct_change()
        returns = df_filtered['returns'].dropna()
        
        # Simple GARCH(1,1) approximation using exponential weighted moving average
        alpha = 0.1  # Weight for current observation
        beta = 0.9   # Weight for previous volatility
        
        garch_vol = np.zeros(len(returns))
        garch_vol[0] = returns.std()
        
        for i in range(1, len(returns)):
            garch_vol[i] = np.sqrt(alpha * returns.iloc[i-1]**2 + beta * garch_vol[i-1]**2)
        
        # Annualize
        garch_vol_annualized = garch_vol * np.sqrt(252)
        
        result = {
            'currency_code': currency_code or 'all',
            'garch_volatility': garch_vol_annualized.tolist(),
            'returns': returns.tolist(),
            'dates': df_filtered['day'].iloc[1:].dt.strftime('%Y-%m-%d').tolist(),  # Skip first date due to returns
            'alpha': alpha,
            'beta': beta,
            'annualized': True
        }
        
        return result
    
    def calculate_historical_volatility(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 30],
                                      currency_code: str = None) -> Dict[str, Any]:
        """
        Calculate historical volatility for different periods
        
        Args:
            df: DataFrame with currency data
            periods: List of periods for volatility calculation
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Historical volatility results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < 2:
            return {
                'currency_code': currency_code or 'all',
                'historical_volatility': {},
                'message': 'Insufficient data for historical volatility calculation'
            }
        
        # Calculate returns
        df_filtered['returns'] = df_filtered['value'].pct_change()
        
        result = {
            'currency_code': currency_code or 'all',
            'historical_volatility': {},
            'dates': df_filtered['day'].dt.strftime('%Y-%m-%d').tolist(),
            'returns': df_filtered['returns'].tolist()
        }
        
        for period in periods:
            if len(df_filtered) >= period:
                # Calculate rolling volatility for each period
                vol = df_filtered['returns'].rolling(window=period).std() * np.sqrt(252)
                result['historical_volatility'][f'vol_{period}d'] = vol.tolist()
            else:
                result['historical_volatility'][f'vol_{period}d'] = [None] * len(df_filtered)
        
        return result
    
    def calculate_volatility_regime(self, df: pd.DataFrame, threshold: float = 0.2,
                                  currency_code: str = None) -> Dict[str, Any]:
        """
        Detect volatility regimes (high/low volatility periods)
        
        Args:
            df: DataFrame with currency data
            threshold: Threshold for high volatility classification
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Volatility regime analysis results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < 20:
            return {
                'currency_code': currency_code or 'all',
                'regimes': [],
                'high_vol_periods': 0,
                'low_vol_periods': 0,
                'current_regime': 'unknown',
                'current_volatility': None,
                'threshold': threshold,
                'message': 'Insufficient data for regime detection'
            }
        
        # Calculate rolling volatility
        df_filtered['returns'] = df_filtered['value'].pct_change()
        rolling_vol = df_filtered['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Classify regimes
        high_vol = rolling_vol > threshold
        low_vol = rolling_vol <= threshold
        
        # Find regime changes
        regime_changes = []
        current_regime = None
        
        for i, (date, vol, is_high) in enumerate(zip(df_filtered['day'], rolling_vol, high_vol)):
            if pd.isna(vol):
                continue
                
            regime = 'high' if is_high else 'low'
            
            if current_regime is None:
                current_regime = regime
                start_date = date
                start_vol = vol
            elif regime != current_regime:
                # Regime change detected
                regime_changes.append({
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': date.strftime('%Y-%m-%d'),
                    'regime': current_regime,
                    'start_volatility': start_vol,
                    'end_volatility': vol,
                    'duration_days': (date - start_date).days
                })
                
                current_regime = regime
                start_date = date
                start_vol = vol
        
        # Add final regime
        if current_regime is not None and len(df_filtered) > 0:
            regime_changes.append({
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': df_filtered['day'].iloc[-1].strftime('%Y-%m-%d'),
                'regime': current_regime,
                'start_volatility': start_vol,
                'end_volatility': rolling_vol.iloc[-1] if not pd.isna(rolling_vol.iloc[-1]) else start_vol,
                'duration_days': (df_filtered['day'].iloc[-1] - start_date).days
            })
        
        result = {
            'currency_code': currency_code or 'all',
            'regimes': regime_changes,
            'high_vol_periods': len([r for r in regime_changes if r['regime'] == 'high']),
            'low_vol_periods': len([r for r in regime_changes if r['regime'] == 'low']),
            'threshold': threshold,
            'current_regime': current_regime if current_regime else 'unknown',
            'current_volatility': rolling_vol.iloc[-1] if not pd.isna(rolling_vol.iloc[-1]) else None
        }
        
        return result
    
    def calculate_volatility_metrics(self, df: pd.DataFrame, currency_code: str = None) -> Dict[str, Any]:
        """
        Calculate comprehensive volatility metrics
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Comprehensive volatility metrics
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < 2:
            return {
                'currency_code': currency_code or 'all',
                'message': 'Insufficient data for volatility metrics calculation'
            }
        
        # Calculate returns
        df_filtered['returns'] = df_filtered['value'].pct_change()
        returns = df_filtered['returns'].dropna()
        
        if len(returns) == 0:
            return {
                'currency_code': currency_code or 'all',
                'message': 'No valid returns data available'
            }
        
        # Basic volatility metrics
        annualized_vol = returns.std() * np.sqrt(252)
        daily_vol = returns.std()
        
        # Calculate different volatility measures
        metrics = {
            'currency_code': currency_code or 'all',
            'annualized_volatility': annualized_vol,
            'daily_volatility': daily_vol,
            'min_volatility': returns.rolling(window=20).std().min() * np.sqrt(252),
            'max_volatility': returns.rolling(window=20).std().max() * np.sqrt(252),
            'volatility_of_volatility': returns.rolling(window=20).std().std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': np.percentile(returns, 5),  # 95% VaR
            'var_99': np.percentile(returns, 1),  # 99% VaR
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),  # 95% CVaR
            'cvar_99': returns[returns <= np.percentile(returns, 1)].mean(),  # 99% CVaR
            'data_points': len(returns),
            'date_range': {
                'start': df_filtered['day'].min().strftime('%Y-%m-%d'),
                'end': df_filtered['day'].max().strftime('%Y-%m-%d')
            }
        }
        
        return metrics
    
    def calculate_volatility_forecast(self, df: pd.DataFrame, forecast_days: int = 30,
                                    currency_code: str = None) -> Dict[str, Any]:
        """
        Simple volatility forecasting using historical patterns
        
        Args:
            df: DataFrame with currency data
            forecast_days: Number of days to forecast
            currency_code: Specific currency to analyze
            
        Returns:
            Dict: Volatility forecast results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        if len(df_filtered) < 30:
            return {
                'currency_code': currency_code or 'all',
                'forecast': [],
                'forecast_dates': [],
                'current_volatility': None,
                'forecast_days': forecast_days,
                'alpha': 0.1,
                'message': 'Insufficient data for volatility forecasting'
            }
        
        # Calculate historical volatility
        df_filtered['returns'] = df_filtered['value'].pct_change()
        rolling_vol = df_filtered['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Simple forecast using exponential smoothing
        alpha = 0.1
        current_vol = rolling_vol.iloc[-1]
        
        if pd.isna(current_vol):
            current_vol = rolling_vol.dropna().iloc[-1] if len(rolling_vol.dropna()) > 0 else 0.2
        
        forecast = []
        for i in range(forecast_days):
            if i == 0:
                forecast_vol = current_vol
            else:
                forecast_vol = alpha * current_vol + (1 - alpha) * forecast[i-1]
            forecast.append(forecast_vol)
        
        # Generate forecast dates
        last_date = df_filtered['day'].iloc[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
        
        result = {
            'currency_code': currency_code or 'all',
            'forecast': forecast,
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'current_volatility': current_vol,
            'forecast_days': forecast_days,
            'alpha': alpha
        }
        
        return result
    
    def analyze_all_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive volatility analysis for all currencies
        
        Args:
            df: DataFrame with currency data
            
        Returns:
            Dict: Comprehensive volatility analysis results
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
                'rolling_volatility': self.calculate_rolling_volatility(df, currency_code=currency),
                'garch_volatility': self.calculate_garch_volatility(df, currency_code=currency),
                'historical_volatility': self.calculate_historical_volatility(df, currency_code=currency),
                'volatility_regime': self.calculate_volatility_regime(df, currency_code=currency),
                'volatility_metrics': self.calculate_volatility_metrics(df, currency_code=currency),
                'volatility_forecast': self.calculate_volatility_forecast(df, currency_code=currency)
            }
            
            results['currencies'][currency] = currency_results
        
        # Calculate overall market volatility
        overall_metrics = self.calculate_volatility_metrics(df)
        results['overall_market'] = {
            'volatility_metrics': overall_metrics
        }
        
        return results
