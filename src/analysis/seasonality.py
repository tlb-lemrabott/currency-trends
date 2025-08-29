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
    
    def analyze_historical_seasonal_peaks(self, df: pd.DataFrame, currency_code: str = None,
                                       min_years: int = 2) -> Dict[str, Any]:
        """
        Analyze historical seasonal peaks to identify consistent patterns
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            min_years: Minimum years of data required for analysis
            
        Returns:
            Dict: Historical seasonal peaks analysis results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        # Add time features
        df_filtered['year'] = df_filtered['day'].dt.year
        df_filtered['month'] = df_filtered['day'].dt.month
        df_filtered['quarter'] = df_filtered['day'].dt.quarter
        
        # Check if we have enough years of data
        unique_years = df_filtered['year'].unique()
        if len(unique_years) < min_years:
            return {
                'currency_code': currency_code or 'all',
                'message': f'Insufficient years of data for historical analysis (minimum {min_years} years required, found {len(unique_years)})'
            }
        
        # Calculate monthly peaks for each year
        yearly_peaks = {}
        monthly_performance = {}
        
        for year in unique_years:
            year_data = df_filtered[df_filtered['year'] == year]
            
            # Find peak month for this year
            monthly_avg = year_data.groupby('month')['value'].mean()
            if len(monthly_avg) > 0:
                peak_month = monthly_avg.idxmax()
                peak_value = monthly_avg.max()
                yearly_peaks[year] = {
                    'peak_month': int(peak_month),
                    'peak_value': peak_value,
                    'monthly_averages': monthly_avg.to_dict()
                }
        
        # Analyze peak month consistency
        peak_months = [data['peak_month'] for data in yearly_peaks.values()]
        peak_month_counts = pd.Series(peak_months).value_counts()
        
        # Find most common peak months
        most_common_peaks = peak_month_counts.head(3).to_dict()
        
        # Calculate monthly performance trends
        all_months = list(range(1, 13))
        for month in all_months:
            month_data = []
            for year in unique_years:
                if year in yearly_peaks and month in yearly_peaks[year]['monthly_averages']:
                    month_data.append(yearly_peaks[year]['monthly_averages'][month])
            
            if month_data:
                monthly_performance[month] = {
                    'average_value': np.mean(month_data),
                    'std_value': np.std(month_data),
                    'trend': self._calculate_trend(month_data),
                    'peak_frequency': peak_month_counts.get(month, 0),
                    'peak_probability': peak_month_counts.get(month, 0) / len(unique_years)
                }
        
        # Identify seasonal trends
        seasonal_trends = self._identify_seasonal_trends(monthly_performance, unique_years)
        
        result = {
            'currency_code': currency_code or 'all',
            'analysis_years': list(unique_years),
            'total_years': len(unique_years),
            'yearly_peaks': yearly_peaks,
            'most_common_peak_months': most_common_peaks,
            'monthly_performance': monthly_performance,
            'seasonal_trends': seasonal_trends,
            'peak_consistency_score': self._calculate_peak_consistency(peak_month_counts, len(unique_years))
        }
        
        return result
    
    def predict_seasonal_peaks(self, df: pd.DataFrame, currency_code: str = None,
                             forecast_years: int = 2) -> Dict[str, Any]:
        """
        Predict future seasonal peaks based on historical patterns
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            forecast_years: Number of years to forecast
            
        Returns:
            Dict: Seasonal peak predictions
        """
        # Get historical analysis
        historical_analysis = self.analyze_historical_seasonal_peaks(df, currency_code)
        
        if 'message' in historical_analysis:
            return {
                'currency_code': currency_code or 'all',
                'message': historical_analysis['message']
            }
        
        # Extract key information
        monthly_performance = historical_analysis['monthly_performance']
        seasonal_trends = historical_analysis['seasonal_trends']
        most_common_peaks = historical_analysis['most_common_peak_months']
        
        # Generate predictions
        predictions = {}
        current_year = pd.Timestamp.now().year
        
        for year in range(current_year + 1, current_year + 1 + forecast_years):
            year_predictions = {}
            
            for month in range(1, 13):
                if month in monthly_performance:
                    month_data = monthly_performance[month]
                    
                    # Calculate predicted value based on trend
                    base_value = month_data['average_value']
                    trend_adjustment = month_data['trend'] * (year - current_year)
                    predicted_value = base_value + trend_adjustment
                    
                    # Calculate confidence based on historical consistency
                    confidence = min(0.95, month_data['peak_probability'] + 0.1)
                    
                    year_predictions[month] = {
                        'predicted_value': predicted_value,
                        'confidence': confidence,
                        'peak_probability': month_data['peak_probability'],
                        'trend_direction': 'increasing' if month_data['trend'] > 0 else 'decreasing' if month_data['trend'] < 0 else 'stable'
                    }
            
            # Find predicted peak months for this year
            month_values = [(month, data['predicted_value']) for month, data in year_predictions.items()]
            month_values.sort(key=lambda x: x[1], reverse=True)
            
            predicted_peaks = []
            for i, (month, value) in enumerate(month_values[:3]):  # Top 3 predicted peaks
                predicted_peaks.append({
                    'rank': i + 1,
                    'month': month,
                    'predicted_value': value,
                    'confidence': year_predictions[month]['confidence'],
                    'peak_probability': year_predictions[month]['peak_probability']
                })
            
            predictions[year] = {
                'predicted_peaks': predicted_peaks,
                'monthly_predictions': year_predictions
            }
        
        # Generate recommendations
        recommendations = self._generate_seasonal_recommendations(
            historical_analysis, predictions, most_common_peaks
        )
        
        result = {
            'currency_code': currency_code or 'all',
            'forecast_years': forecast_years,
            'historical_analysis': historical_analysis,
            'predictions': predictions,
            'recommendations': recommendations,
            'confidence_factors': {
                'data_quality': len(historical_analysis['analysis_years']) / 5,  # Normalize to 5 years
                'pattern_consistency': historical_analysis['peak_consistency_score'],
                'trend_strength': self._calculate_trend_strength(seasonal_trends)
            }
        }
        
        return result
    
    def analyze_seasonal_momentum(self, df: pd.DataFrame, currency_code: str = None,
                                lookback_years: int = 3) -> Dict[str, Any]:
        """
        Analyze seasonal momentum - how seasonal patterns are changing over time
        
        Args:
            df: DataFrame with currency data
            currency_code: Specific currency to analyze
            lookback_years: Number of recent years to analyze for momentum
            
        Returns:
            Dict: Seasonal momentum analysis results
        """
        if currency_code:
            df_filtered = df[df['currency_code'] == currency_code].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered = df_filtered.sort_values('day').reset_index(drop=True)
        
        # Add time features
        df_filtered['year'] = df_filtered['day'].dt.year
        df_filtered['month'] = df_filtered['day'].dt.month
        
        # Get recent years
        unique_years = sorted(df_filtered['year'].unique())
        recent_years = unique_years[-lookback_years:] if len(unique_years) >= lookback_years else unique_years
        
        if len(recent_years) < 2:
            return {
                'currency_code': currency_code or 'all',
                'message': f'Insufficient recent data for momentum analysis (minimum 2 years required, found {len(recent_years)})'
            }
        
        # Calculate monthly momentum for each year
        momentum_data = {}
        for year in recent_years:
            year_data = df_filtered[df_filtered['year'] == year]
            monthly_avg = year_data.groupby('month')['value'].mean()
            momentum_data[year] = monthly_avg.to_dict()
        
        # Calculate momentum indicators
        momentum_indicators = {}
        for month in range(1, 13):
            month_values = []
            for year in recent_years:
                if month in momentum_data[year]:
                    month_values.append(momentum_data[year][month])
            
            if len(month_values) >= 2:
                # Calculate momentum (rate of change)
                momentum = (month_values[-1] - month_values[0]) / len(month_values) if len(month_values) > 1 else 0
                
                # Calculate acceleration (change in momentum)
                acceleration = 0
                if len(month_values) >= 3:
                    momentum_changes = [month_values[i+1] - month_values[i] for i in range(len(month_values)-1)]
                    acceleration = np.mean(momentum_changes[1:]) - np.mean(momentum_changes[:-1]) if len(momentum_changes) > 1 else 0
                
                momentum_indicators[month] = {
                    'values': month_values,
                    'momentum': momentum,
                    'acceleration': acceleration,
                    'trend': 'increasing' if momentum > 0 else 'decreasing' if momentum < 0 else 'stable',
                    'acceleration_trend': 'accelerating' if acceleration > 0 else 'decelerating' if acceleration < 0 else 'stable'
                }
        
        # Identify emerging patterns
        emerging_patterns = self._identify_emerging_patterns(momentum_indicators, recent_years)
        
        result = {
            'currency_code': currency_code or 'all',
            'analysis_years': recent_years,
            'lookback_years': lookback_years,
            'momentum_data': momentum_data,
            'momentum_indicators': momentum_indicators,
            'emerging_patterns': emerging_patterns,
            'overall_momentum': self._calculate_overall_momentum(momentum_indicators)
        }
        
        return result
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _identify_seasonal_trends(self, monthly_performance: Dict, years: List[int]) -> Dict[str, Any]:
        """Identify seasonal trends from monthly performance data"""
        trends = {
            'strongest_months': [],
            'weakest_months': [],
            'improving_months': [],
            'declining_months': [],
            'stable_months': []
        }
        
        # Sort months by average performance
        month_avg_performance = [(month, data['average_value']) for month, data in monthly_performance.items()]
        month_avg_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Identify strongest and weakest months
        trends['strongest_months'] = [month for month, _ in month_avg_performance[:3]]
        trends['weakest_months'] = [month for month, _ in month_avg_performance[-3:]]
        
        # Identify improving and declining months
        for month, data in monthly_performance.items():
            if data['trend'] > 0.1:  # Significant positive trend
                trends['improving_months'].append(month)
            elif data['trend'] < -0.1:  # Significant negative trend
                trends['declining_months'].append(month)
            else:
                trends['stable_months'].append(month)
        
        return trends
    
    def _calculate_peak_consistency(self, peak_counts: pd.Series, total_years: int) -> float:
        """Calculate consistency score for peak months"""
        if total_years == 0:
            return 0.0
        
        # Calculate how concentrated the peaks are
        max_peaks = peak_counts.max() if len(peak_counts) > 0 else 0
        consistency_score = max_peaks / total_years
        
        return min(consistency_score, 1.0)
    
    def _identify_emerging_patterns(self, momentum_indicators: Dict, years: List[int]) -> List[Dict[str, Any]]:
        """Identify emerging seasonal patterns"""
        emerging_patterns = []
        
        for month, data in momentum_indicators.items():
            # Check for strong momentum
            if abs(data['momentum']) > 0.5:  # Significant momentum
                pattern = {
                    'month': month,
                    'type': 'strong_momentum',
                    'direction': data['trend'],
                    'strength': abs(data['momentum']),
                    'description': f"Month {month} showing strong {data['trend']} momentum"
                }
                emerging_patterns.append(pattern)
            
            # Check for acceleration
            if abs(data['acceleration']) > 0.2:  # Significant acceleration
                pattern = {
                    'month': month,
                    'type': 'acceleration',
                    'direction': data['acceleration_trend'],
                    'strength': abs(data['acceleration']),
                    'description': f"Month {month} showing {data['acceleration_trend']} pattern"
                }
                emerging_patterns.append(pattern)
        
        return emerging_patterns
    
    def _calculate_overall_momentum(self, momentum_indicators: Dict) -> Dict[str, Any]:
        """Calculate overall momentum across all months"""
        if not momentum_indicators:
            return {'overall_trend': 'unknown', 'strength': 0.0}
        
        # Calculate weighted average momentum
        total_momentum = 0
        total_weight = 0
        
        for month, data in momentum_indicators.items():
            weight = len(data['values'])  # Weight by number of observations
            total_momentum += data['momentum'] * weight
            total_weight += weight
        
        overall_momentum = total_momentum / total_weight if total_weight > 0 else 0
        
        return {
            'overall_trend': 'increasing' if overall_momentum > 0.1 else 'decreasing' if overall_momentum < -0.1 else 'stable',
            'strength': abs(overall_momentum),
            'value': overall_momentum
        }
    
    def _calculate_trend_strength(self, seasonal_trends: Dict) -> float:
        """Calculate overall trend strength"""
        improving_count = len(seasonal_trends.get('improving_months', []))
        declining_count = len(seasonal_trends.get('declining_months', []))
        total_months = improving_count + declining_count
        
        if total_months == 0:
            return 0.0
        
        # Calculate strength based on consistency of trends
        strength = max(improving_count, declining_count) / total_months
        return strength
    
    def _generate_seasonal_recommendations(self, historical_analysis: Dict, 
                                        predictions: Dict, most_common_peaks: Dict) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on seasonal analysis"""
        recommendations = []
        
        # Recommendation 1: Most reliable peak months
        if most_common_peaks:
            top_peak_month = max(most_common_peaks.items(), key=lambda x: x[1])
            recommendations.append({
                'type': 'peak_timing',
                'priority': 'high',
                'description': f"Month {top_peak_month[0]} has been the peak month in {top_peak_month[1]} out of {len(historical_analysis['analysis_years'])} years",
                'action': f"Consider positioning for peak performance in month {top_peak_month[0]}",
                'confidence': min(0.95, top_peak_month[1] / len(historical_analysis['analysis_years']))
            })
        
        # Recommendation 2: Emerging patterns
        if 'seasonal_trends' in historical_analysis:
            improving_months = historical_analysis['seasonal_trends'].get('improving_months', [])
            if improving_months:
                recommendations.append({
                    'type': 'trend_analysis',
                    'priority': 'medium',
                    'description': f"Months {improving_months} are showing improving seasonal trends",
                    'action': "Monitor these months for potential breakout opportunities",
                    'confidence': 0.7
                })
        
        # Recommendation 3: Prediction confidence
        if predictions:
            next_year = list(predictions.keys())[0]
            top_prediction = predictions[next_year]['predicted_peaks'][0]
            if top_prediction['confidence'] > 0.6:
                recommendations.append({
                    'type': 'forecast',
                    'priority': 'medium',
                    'description': f"Month {top_prediction['month']} is predicted to be the peak in {next_year}",
                    'action': f"Prepare for potential peak in month {top_prediction['month']} of {next_year}",
                    'confidence': top_prediction['confidence']
                })
        
        return recommendations
    
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
                'stationarity': self.test_stationarity(df, currency_code=currency),
                'historical_peaks': self.analyze_historical_seasonal_peaks(df, currency_code=currency),
                'seasonal_predictions': self.predict_seasonal_peaks(df, currency_code=currency),
                'seasonal_momentum': self.analyze_seasonal_momentum(df, currency_code=currency)
            }
            
            results[currency] = currency_results
        
        # Calculate overall market seasonality
        overall_seasonal = self.detect_seasonal_patterns(df)
        results['overall_market'] = {
            'seasonal_patterns': overall_seasonal
        }
        
        return results
