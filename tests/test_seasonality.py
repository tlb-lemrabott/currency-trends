"""
Unit tests for Seasonality Analysis Module

Tests for the SeasonalityAnalyzer class and its methods.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analysis.seasonality import SeasonalityAnalyzer


class TestSeasonalityAnalyzer:
    """Test cases for SeasonalityAnalyzer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample currency data with seasonal patterns for testing"""
        dates = pd.date_range(start='2020-01-01', periods=365, freq='D')  # Full year of data
        
        # Create sample data with known seasonal patterns
        usd_data = []
        eur_data = []
        
        for i, date in enumerate(dates):
            # USD: Add seasonal component (monthly pattern)
            month = date.month
            seasonal_component = 5 * np.sin(2 * np.pi * month / 12)  # Monthly seasonality
            trend_component = i * 0.1  # Upward trend
            noise = np.random.normal(0, 1)
            usd_value = 100 + trend_component + seasonal_component + noise
            
            usd_data.append({
                'day': date,
                'currency_code': 'USD',
                'value': usd_value
            })
            
            # EUR: Add different seasonal pattern
            seasonal_component = 3 * np.cos(2 * np.pi * month / 12)  # Different phase
            trend_component = i * 0.05  # Slower trend
            noise = np.random.normal(0, 0.8)
            eur_value = 120 + trend_component + seasonal_component + noise
            
            eur_data.append({
                'day': date,
                'currency_code': 'EUR',
                'value': eur_value
            })
        
        return pd.DataFrame(usd_data + eur_data)
    
    @pytest.fixture
    def analyzer(self):
        """Create SeasonalityAnalyzer instance"""
        return SeasonalityAnalyzer()
    
    def test_detect_seasonal_patterns(self, analyzer, sample_data):
        """Test seasonal pattern detection"""
        result = analyzer.detect_seasonal_patterns(sample_data, currency_code='USD', period=12)
        
        assert result['currency_code'] == 'USD'
        assert result['period'] == 12
        assert 'seasonal_strength' in result
        assert 'trend' in result
        assert 'seasonal' in result
        assert 'residual' in result
        assert 'seasonal_peaks' in result
        assert 'data_points' in result
        assert 'date_range' in result
        
        assert result['data_points'] == 365
        assert 'start' in result['date_range']
        assert 'end' in result['date_range']
        assert 0 <= result['seasonal_strength'] <= 1
    
    def test_detect_seasonal_patterns_insufficient_data(self, analyzer):
        """Test seasonal pattern detection with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        small_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.detect_seasonal_patterns(small_data, currency_code='USD', period=12)
        
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_analyze_monthly_patterns(self, analyzer, sample_data):
        """Test monthly pattern analysis"""
        result = analyzer.analyze_monthly_patterns(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert 'monthly_stats' in result
        assert 'strongest_month' in result
        assert 'weakest_month' in result
        assert 'seasonal_variation' in result
        assert 'overall_mean' in result
        assert 'data_points' in result
        
        assert len(result['monthly_stats']) == 12  # 12 months
        assert result['data_points'] == 365
        assert result['seasonal_variation'] >= 0
        assert result['overall_mean'] > 0
        
        # Check strongest and weakest months
        assert 'month' in result['strongest_month']
        assert 'seasonal_index' in result['strongest_month']
        assert 'month' in result['weakest_month']
        assert 'seasonal_index' in result['weakest_month']
    
    def test_analyze_monthly_patterns_insufficient_data(self, analyzer):
        """Test monthly pattern analysis with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        small_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.analyze_monthly_patterns(small_data, currency_code='USD')
        
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_analyze_quarterly_patterns(self, analyzer, sample_data):
        """Test quarterly pattern analysis"""
        result = analyzer.analyze_quarterly_patterns(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert 'quarterly_stats' in result
        assert 'strongest_quarter' in result
        assert 'weakest_quarter' in result
        assert 'seasonal_variation' in result
        assert 'overall_mean' in result
        assert 'data_points' in result
        
        assert len(result['quarterly_stats']) == 4  # 4 quarters
        assert result['data_points'] == 365
        assert result['seasonal_variation'] >= 0
        assert result['overall_mean'] > 0
        
        # Check strongest and weakest quarters
        assert 'quarter' in result['strongest_quarter']
        assert 'seasonal_index' in result['strongest_quarter']
        assert 'quarter' in result['weakest_quarter']
        assert 'seasonal_index' in result['weakest_quarter']
    
    def test_analyze_quarterly_patterns_insufficient_data(self, analyzer):
        """Test quarterly pattern analysis with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        small_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.analyze_quarterly_patterns(small_data, currency_code='USD')
        
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_analyze_weekly_patterns(self, analyzer, sample_data):
        """Test weekly pattern analysis"""
        result = analyzer.analyze_weekly_patterns(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert 'daily_stats' in result
        assert 'strongest_day' in result
        assert 'weakest_day' in result
        assert 'seasonal_variation' in result
        assert 'overall_mean' in result
        assert 'data_points' in result
        
        assert len(result['daily_stats']) == 7  # 7 days of week
        assert result['data_points'] == 365
        assert result['seasonal_variation'] >= 0
        assert result['overall_mean'] > 0
        
        # Check strongest and weakest days
        assert 'day_of_week' in result['strongest_day']
        assert 'day_name' in result['strongest_day']
        assert 'seasonal_index' in result['strongest_day']
        assert 'day_of_week' in result['weakest_day']
        assert 'day_name' in result['weakest_day']
        assert 'seasonal_index' in result['weakest_day']
    
    def test_analyze_weekly_patterns_insufficient_data(self, analyzer):
        """Test weekly pattern analysis with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        small_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.analyze_weekly_patterns(small_data, currency_code='USD')
        
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_detect_periodicity(self, analyzer, sample_data):
        """Test periodicity detection"""
        result = analyzer.detect_periodicity(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert 'dominant_frequencies' in result
        assert 'dominant_powers' in result
        assert 'periods' in result
        assert 'significant_peaks' in result
        assert 'significance_threshold' in result
        assert 'data_points' in result
        
        assert len(result['dominant_frequencies']) == 3  # Top 3 frequencies
        assert len(result['dominant_powers']) == 3
        assert len(result['periods']) == 3
        assert result['data_points'] == 365
        assert result['significance_threshold'] >= 0
    
    def test_detect_periodicity_insufficient_data(self, analyzer):
        """Test periodicity detection with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        small_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.detect_periodicity(small_data, currency_code='USD')
        
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_test_stationarity(self, analyzer, sample_data):
        """Test stationarity testing"""
        result = analyzer.test_stationarity(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert 'adf_statistic' in result
        assert 'p_value' in result
        assert 'critical_values' in result
        assert 'is_stationary' in result
        assert 'significance_level' in result
        assert 'data_points' in result
        
        assert result['data_points'] == 365
        assert result['significance_level'] == 0.05
        assert isinstance(result['is_stationary'], (bool, np.bool_))
        assert isinstance(result['p_value'], float)
    
    def test_test_stationarity_insufficient_data(self, analyzer):
        """Test stationarity testing with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        small_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.test_stationarity(small_data, currency_code='USD')
        
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_calculate_seasonal_strength(self, analyzer):
        """Test seasonal strength calculation"""
        # Create sample data
        ts_data = pd.Series([100 + 5*np.sin(2*np.pi*i/12) + np.random.normal(0,1) for i in range(50)])
        seasonal = pd.Series([5*np.sin(2*np.pi*i/12) for i in range(50)])
        residual = pd.Series([np.random.normal(0,1) for i in range(50)])
        
        strength = analyzer._calculate_seasonal_strength(ts_data, seasonal, residual)
        
        assert 0 <= strength <= 1
        assert strength > 0  # Should have some seasonal strength
    
    def test_calculate_seasonal_strength_zero_variance(self, analyzer):
        """Test seasonal strength calculation with zero variance"""
        ts_data = pd.Series([100] * 10)
        seasonal = pd.Series([0] * 10)
        residual = pd.Series([0] * 10)
        
        strength = analyzer._calculate_seasonal_strength(ts_data, seasonal, residual)
        
        assert strength == 0.0
    
    def test_find_seasonal_peaks(self, analyzer):
        """Test seasonal peaks detection"""
        # Create seasonal data with known peaks
        seasonal = pd.Series([np.sin(2*np.pi*i/12) for i in range(24)])
        
        peaks = analyzer._find_seasonal_peaks(seasonal, 12)
        
        assert len(peaks) == 12
        for peak in peaks:
            assert 'period_position' in peak
            assert 'max_value' in peak
            assert 'min_value' in peak
            assert 'amplitude' in peak
            assert peak['amplitude'] >= 0
    
    def test_find_seasonal_peaks_empty_data(self, analyzer):
        """Test seasonal peaks detection with empty data"""
        seasonal = pd.Series([])
        
        peaks = analyzer._find_seasonal_peaks(seasonal, 12)
        
        assert len(peaks) == 0
    
    def test_analyze_all_seasonality(self, analyzer, sample_data):
        """Test comprehensive seasonality analysis"""
        result = analyzer.analyze_all_seasonality(sample_data)
        
        assert 'summary' in result
        assert 'overall_market' in result
        assert 'USD' in result
        assert 'EUR' in result
        
        # Check summary
        assert result['summary']['total_currencies'] == 2
        assert 'analysis_date' in result['summary']
        
        # Check each currency has all analysis types
        for currency in ['USD', 'EUR']:
            currency_result = result[currency]
            assert 'seasonal_patterns' in currency_result
            assert 'monthly_patterns' in currency_result
            assert 'quarterly_patterns' in currency_result
            assert 'weekly_patterns' in currency_result
            assert 'periodicity' in currency_result
            assert 'stationarity' in currency_result
        
        # Check overall market
        assert 'seasonal_patterns' in result['overall_market']
    
    def test_seasonal_patterns_different_currencies(self, analyzer, sample_data):
        """Test that different currencies show different seasonal patterns"""
        usd_monthly = analyzer.analyze_monthly_patterns(sample_data, currency_code='USD')
        eur_monthly = analyzer.analyze_monthly_patterns(sample_data, currency_code='EUR')
        
        # Should have different seasonal variations due to different patterns
        assert usd_monthly['seasonal_variation'] != eur_monthly['seasonal_variation']
    
    def test_empty_dataframe(self, analyzer):
        """Test behavior with empty DataFrame"""
        empty_df = pd.DataFrame(columns=['day', 'currency_code', 'value'])
        
        result = analyzer.detect_seasonal_patterns(empty_df)
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_missing_currency_code(self, analyzer, sample_data):
        """Test behavior when currency code doesn't exist"""
        result = analyzer.detect_seasonal_patterns(sample_data, currency_code='NONEXISTENT')
        
        assert result['currency_code'] == 'NONEXISTENT'
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_seasonal_decomposition_error_handling(self, analyzer):
        """Test error handling in seasonal decomposition"""
        # Create data that might cause decomposition issues
        dates = pd.date_range(start='2024-01-01', periods=15, freq='D')
        problematic_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 if i % 2 == 0 else np.nan  # Alternating NaN values
        } for i, date in enumerate(dates)])
        
        result = analyzer.detect_seasonal_patterns(problematic_data, currency_code='USD', period=12)
        
        # Should handle gracefully
        assert 'message' in result or 'seasonal_strength' in result
