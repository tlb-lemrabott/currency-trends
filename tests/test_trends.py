"""
Unit tests for Trend Analysis Module

Tests for the TrendAnalyzer class and its methods.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analysis.trends import TrendAnalyzer


class TestTrendAnalyzer:
    """Test cases for TrendAnalyzer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample currency data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # Create sample data with known trends
        usd_data = []
        eur_data = []
        
        for i, date in enumerate(dates):
            # USD: increasing trend
            usd_value = 100 + (i * 0.5) + np.random.normal(0, 1)
            usd_data.append({
                'day': date,
                'currency_code': 'USD',
                'value': usd_value
            })
            
            # EUR: decreasing trend
            eur_value = 120 - (i * 0.3) + np.random.normal(0, 1)
            eur_data.append({
                'day': date,
                'currency_code': 'EUR',
                'value': eur_value
            })
        
        return pd.DataFrame(usd_data + eur_data)
    
    @pytest.fixture
    def analyzer(self):
        """Create TrendAnalyzer instance"""
        return TrendAnalyzer()
    
    def test_calculate_linear_trend_increasing(self, analyzer, sample_data):
        """Test linear trend calculation for increasing trend"""
        result = analyzer.calculate_linear_trend(sample_data, 'USD')
        
        assert result['currency_code'] == 'USD'
        assert result['trend_direction'] == 'increasing'
        assert result['slope'] > 0
        assert 0 <= result['r2_score'] <= 1
        assert result['total_change_percent'] > 0
        assert result['data_points'] == 30
        assert len(result['predictions']) == 30
        assert len(result['actual_values']) == 30
        assert len(result['dates']) == 30
    
    def test_calculate_linear_trend_decreasing(self, analyzer, sample_data):
        """Test linear trend calculation for decreasing trend"""
        result = analyzer.calculate_linear_trend(sample_data, 'EUR')
        
        assert result['currency_code'] == 'EUR'
        assert result['trend_direction'] == 'decreasing'
        assert result['slope'] < 0
        assert 0 <= result['r2_score'] <= 1
        assert result['total_change_percent'] < 0
        assert result['data_points'] == 30
    
    def test_calculate_linear_trend_all_currencies(self, analyzer, sample_data):
        """Test linear trend calculation for all currencies"""
        result = analyzer.calculate_linear_trend(sample_data)
        
        assert result['currency_code'] == 'all'
        assert 'slope' in result
        assert 'r2_score' in result
        assert 'trend_direction' in result
        assert result['data_points'] == 60  # 30 USD + 30 EUR
    
    def test_calculate_polynomial_trend(self, analyzer, sample_data):
        """Test polynomial trend calculation"""
        result = analyzer.calculate_polynomial_trend(sample_data, degree=2, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert result['degree'] == 2
        assert len(result['coefficients']) == 3  # degree + 1
        assert 0 <= result['r2_score'] <= 1
        assert result['data_points'] == 30
        assert len(result['predictions']) == 30
    
    def test_calculate_polynomial_trend_degree_3(self, analyzer, sample_data):
        """Test polynomial trend calculation with degree 3"""
        result = analyzer.calculate_polynomial_trend(sample_data, degree=3, currency_code='USD')
        
        assert result['degree'] == 3
        assert len(result['coefficients']) == 4  # degree + 1
    
    def test_calculate_moving_averages(self, analyzer, sample_data):
        """Test moving averages calculation"""
        result = analyzer.calculate_moving_averages(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert len(result['dates']) == 30
        assert len(result['actual_values']) == 30
        assert 'ma_7' in result['moving_averages']
        assert 'ma_14' in result['moving_averages']
        assert 'ma_30' in result['moving_averages']
        assert len(result['moving_averages']['ma_7']) == 30
    
    def test_calculate_moving_averages_custom_windows(self, analyzer, sample_data):
        """Test moving averages with custom windows"""
        custom_windows = [5, 10, 15]
        result = analyzer.calculate_moving_averages(sample_data, windows=custom_windows, currency_code='USD')
        
        assert 'ma_5' in result['moving_averages']
        assert 'ma_10' in result['moving_averages']
        assert 'ma_15' in result['moving_averages']
    
    def test_calculate_exponential_moving_average(self, analyzer, sample_data):
        """Test exponential moving average calculation"""
        result = analyzer.calculate_exponential_moving_average(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert len(result['dates']) == 30
        assert len(result['actual_values']) == 30
        assert 'ema_12' in result['exponential_moving_averages']
        assert 'ema_26' in result['exponential_moving_averages']
        assert len(result['exponential_moving_averages']['ema_12']) == 30
    
    def test_calculate_exponential_moving_average_custom_spans(self, analyzer, sample_data):
        """Test exponential moving average with custom spans"""
        custom_spans = [5, 10]
        result = analyzer.calculate_exponential_moving_average(sample_data, spans=custom_spans, currency_code='USD')
        
        assert 'ema_5' in result['exponential_moving_averages']
        assert 'ema_10' in result['exponential_moving_averages']
    
    def test_detect_trend_changes_sufficient_data(self, analyzer, sample_data):
        """Test trend change detection with sufficient data"""
        result = analyzer.detect_trend_changes(sample_data, window=5, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert 'trend_changes' in result
        assert 'total_changes' in result
        assert 'bullish_changes' in result
        assert 'bearish_changes' in result
        assert result['total_changes'] >= 0
        assert result['bullish_changes'] >= 0
        assert result['bearish_changes'] >= 0
    
    def test_detect_trend_changes_insufficient_data(self, analyzer):
        """Test trend change detection with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        small_data = pd.DataFrame([
            {'day': date, 'currency_code': 'USD', 'value': 100 + i}
            for i, date in enumerate(dates)
        ])
        
        result = analyzer.detect_trend_changes(small_data, window=10, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert result['total_changes'] == 0
        assert result['bullish_changes'] == 0
        assert result['bearish_changes'] == 0
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_calculate_trend_strength(self, analyzer, sample_data):
        """Test trend strength calculation"""
        result = analyzer.calculate_trend_strength(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert 'positive_days' in result
        assert 'negative_days' in result
        assert 'total_days' in result
        assert 'trend_consistency' in result
        assert 'avg_daily_change' in result
        assert 'avg_daily_change_pct' in result
        assert 'momentum' in result
        assert 'strength_category' in result
        assert 'trend_direction' in result
        
        assert result['total_days'] >= 0
        assert 0 <= result['trend_consistency'] <= 1
        assert result['strength_category'] in ['weak', 'moderate', 'strong']
        assert result['trend_direction'] in ['bullish', 'bearish', 'neutral']
    
    def test_calculate_trend_strength_all_currencies(self, analyzer, sample_data):
        """Test trend strength calculation for all currencies"""
        result = analyzer.calculate_trend_strength(sample_data)
        
        assert result['currency_code'] == 'all'
        assert 'positive_days' in result
        assert 'negative_days' in result
        assert 'trend_consistency' in result
    
    def test_analyze_all_trends(self, analyzer, sample_data):
        """Test comprehensive trend analysis"""
        result = analyzer.analyze_all_trends(sample_data)
        
        assert 'currencies' in result
        assert 'summary' in result
        assert 'overall_market' in result
        
        # Check summary
        assert result['summary']['total_currencies'] == 2
        assert 'analysis_date' in result['summary']
        
        # Check currencies
        assert 'USD' in result['currencies']
        assert 'EUR' in result['currencies']
        
        # Check each currency has all analysis types
        for currency in ['USD', 'EUR']:
            currency_result = result['currencies'][currency]
            assert 'linear_trend' in currency_result
            assert 'polynomial_trend' in currency_result
            assert 'moving_averages' in currency_result
            assert 'ema' in currency_result
            assert 'trend_changes' in currency_result
            assert 'trend_strength' in currency_result
        
        # Check overall market
        assert 'linear_trend' in result['overall_market']
        assert 'trend_strength' in result['overall_market']
    
    def test_empty_dataframe(self, analyzer):
        """Test behavior with empty DataFrame"""
        empty_df = pd.DataFrame(columns=['day', 'currency_code', 'value'])
        
        # Should handle empty DataFrame gracefully
        result = analyzer.calculate_linear_trend(empty_df)
        assert result['data_points'] == 0
    
    def test_single_data_point(self, analyzer):
        """Test behavior with single data point"""
        single_df = pd.DataFrame([{
            'day': pd.Timestamp('2024-01-01'),
            'currency_code': 'USD',
            'value': 100.0
        }])
        
        result = analyzer.calculate_linear_trend(single_df, 'USD')
        assert result['data_points'] == 1
        assert result['slope'] == 0  # No trend with single point
    
    def test_missing_currency_code(self, analyzer, sample_data):
        """Test behavior when currency code doesn't exist"""
        result = analyzer.calculate_linear_trend(sample_data, 'NONEXISTENT')
        assert result['data_points'] == 0
    
    def test_trend_strength_categories(self, analyzer):
        """Test trend strength categorization"""
        # Create data with different trend consistencies
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        
        # Strong bullish trend
        strong_bullish = pd.DataFrame([
            {'day': date, 'currency_code': 'USD', 'value': 100 + i + np.random.normal(0, 0.1)}
            for i, date in enumerate(dates)
        ])
        
        # Weak trend (random)
        weak_trend = pd.DataFrame([
            {'day': date, 'currency_code': 'EUR', 'value': 100 + np.random.normal(0, 5)}
            for i, date in enumerate(dates)
        ])
        
        strong_result = analyzer.calculate_trend_strength(strong_bullish, 'USD')
        weak_result = analyzer.calculate_trend_strength(weak_trend, 'EUR')
        
        assert strong_result['strength_category'] == 'strong'
        # The weak trend might be categorized as moderate due to random variation
        assert weak_result['strength_category'] in ['weak', 'moderate']
        assert strong_result['trend_direction'] == 'bullish'
