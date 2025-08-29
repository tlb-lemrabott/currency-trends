"""
Unit tests for Volatility Analysis Module

Tests for the VolatilityAnalyzer class and its methods.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analysis.volatility import VolatilityAnalyzer


class TestVolatilityAnalyzer:
    """Test cases for VolatilityAnalyzer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample currency data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Create sample data with known volatility patterns
        usd_data = []
        eur_data = []
        
        for i, date in enumerate(dates):
            # USD: high volatility
            usd_value = 100 + (i * 0.1) + np.random.normal(0, 2)
            usd_data.append({
                'day': date,
                'currency_code': 'USD',
                'value': usd_value
            })
            
            # EUR: low volatility
            eur_value = 120 + (i * 0.05) + np.random.normal(0, 0.5)
            eur_data.append({
                'day': date,
                'currency_code': 'EUR',
                'value': eur_value
            })
        
        return pd.DataFrame(usd_data + eur_data)
    
    @pytest.fixture
    def analyzer(self):
        """Create VolatilityAnalyzer instance"""
        return VolatilityAnalyzer()
    
    def test_calculate_rolling_volatility(self, analyzer, sample_data):
        """Test rolling volatility calculation"""
        result = analyzer.calculate_rolling_volatility(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert len(result['volatility']) == 50
        assert len(result['returns']) == 50
        assert len(result['dates']) == 50
        assert result['window'] == 20
        assert result['annualized'] is True
        
        # Check that volatility values are reasonable
        vol_values = [v for v in result['volatility'] if not pd.isna(v)]
        assert len(vol_values) > 0
        assert all(v >= 0 for v in vol_values)
    
    def test_calculate_rolling_volatility_custom_window(self, analyzer, sample_data):
        """Test rolling volatility with custom window"""
        result = analyzer.calculate_rolling_volatility(sample_data, window=10, currency_code='USD')
        
        assert result['window'] == 10
        assert len(result['volatility']) == 50
    
    def test_calculate_rolling_volatility_insufficient_data(self, analyzer):
        """Test rolling volatility with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=1, freq='D')
        small_data = pd.DataFrame([{
            'day': dates[0],
            'currency_code': 'USD',
            'value': 100.0
        }])
        
        result = analyzer.calculate_rolling_volatility(small_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert len(result['volatility']) == 0
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_calculate_garch_volatility(self, analyzer, sample_data):
        """Test GARCH volatility calculation"""
        result = analyzer.calculate_garch_volatility(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert len(result['garch_volatility']) > 0
        assert len(result['returns']) > 0
        assert len(result['dates']) > 0
        assert 'alpha' in result
        assert 'beta' in result
        assert result['annualized'] is True
        
        # Check that GARCH volatility values are reasonable
        garch_values = [v for v in result['garch_volatility'] if not pd.isna(v)]
        assert len(garch_values) > 0
        assert all(v >= 0 for v in garch_values)
    
    def test_calculate_garch_volatility_insufficient_data(self, analyzer):
        """Test GARCH volatility with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        small_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.calculate_garch_volatility(small_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert len(result['garch_volatility']) == 0
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_calculate_historical_volatility(self, analyzer, sample_data):
        """Test historical volatility calculation"""
        result = analyzer.calculate_historical_volatility(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert 'historical_volatility' in result
        assert len(result['dates']) == 50
        assert len(result['returns']) == 50
        
        # Check that different period volatilities are calculated
        hist_vol = result['historical_volatility']
        assert 'vol_5d' in hist_vol
        assert 'vol_10d' in hist_vol
        assert 'vol_20d' in hist_vol
        assert 'vol_30d' in hist_vol
    
    def test_calculate_historical_volatility_custom_periods(self, analyzer, sample_data):
        """Test historical volatility with custom periods"""
        custom_periods = [7, 14]
        result = analyzer.calculate_historical_volatility(sample_data, periods=custom_periods, currency_code='USD')
        
        hist_vol = result['historical_volatility']
        assert 'vol_7d' in hist_vol
        assert 'vol_14d' in hist_vol
    
    def test_calculate_volatility_regime(self, analyzer, sample_data):
        """Test volatility regime detection"""
        result = analyzer.calculate_volatility_regime(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert 'regimes' in result
        assert 'high_vol_periods' in result
        assert 'low_vol_periods' in result
        assert 'threshold' in result
        assert 'current_regime' in result
        assert 'current_volatility' in result
        
        assert result['high_vol_periods'] >= 0
        assert result['low_vol_periods'] >= 0
        assert result['threshold'] == 0.2
    
    def test_calculate_volatility_regime_custom_threshold(self, analyzer, sample_data):
        """Test volatility regime with custom threshold"""
        result = analyzer.calculate_volatility_regime(sample_data, threshold=0.1, currency_code='USD')
        
        assert result['threshold'] == 0.1
    
    def test_calculate_volatility_regime_insufficient_data(self, analyzer):
        """Test volatility regime with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        small_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.calculate_volatility_regime(small_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert len(result['regimes']) == 0
        assert result['high_vol_periods'] == 0
        assert result['low_vol_periods'] == 0
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_calculate_volatility_metrics(self, analyzer, sample_data):
        """Test volatility metrics calculation"""
        result = analyzer.calculate_volatility_metrics(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert 'annualized_volatility' in result
        assert 'daily_volatility' in result
        assert 'min_volatility' in result
        assert 'max_volatility' in result
        assert 'volatility_of_volatility' in result
        assert 'skewness' in result
        assert 'kurtosis' in result
        assert 'var_95' in result
        assert 'var_99' in result
        assert 'cvar_95' in result
        assert 'cvar_99' in result
        assert 'data_points' in result
        assert 'date_range' in result
        
        # Check that metrics are reasonable
        assert result['annualized_volatility'] > 0
        assert result['daily_volatility'] > 0
        assert result['data_points'] > 0
        assert 'start' in result['date_range']
        assert 'end' in result['date_range']
    
    def test_calculate_volatility_metrics_insufficient_data(self, analyzer):
        """Test volatility metrics with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=1, freq='D')
        small_data = pd.DataFrame([{
            'day': dates[0],
            'currency_code': 'USD',
            'value': 100.0
        }])
        
        result = analyzer.calculate_volatility_metrics(small_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_calculate_volatility_forecast(self, analyzer, sample_data):
        """Test volatility forecasting"""
        result = analyzer.calculate_volatility_forecast(sample_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert len(result['forecast']) == 30
        assert len(result['forecast_dates']) == 30
        assert 'current_volatility' in result
        assert 'forecast_days' in result
        assert 'alpha' in result
        
        assert result['forecast_days'] == 30
        assert result['alpha'] == 0.1
        
        # Check that forecast values are reasonable
        assert all(v >= 0 for v in result['forecast'])
    
    def test_calculate_volatility_forecast_custom_days(self, analyzer, sample_data):
        """Test volatility forecasting with custom forecast days"""
        result = analyzer.calculate_volatility_forecast(sample_data, forecast_days=10, currency_code='USD')
        
        assert len(result['forecast']) == 10
        assert len(result['forecast_dates']) == 10
        assert result['forecast_days'] == 10
    
    def test_calculate_volatility_forecast_insufficient_data(self, analyzer):
        """Test volatility forecasting with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        small_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.calculate_volatility_forecast(small_data, currency_code='USD')
        
        assert result['currency_code'] == 'USD'
        assert len(result['forecast']) == 0
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_analyze_all_volatility(self, analyzer, sample_data):
        """Test comprehensive volatility analysis"""
        result = analyzer.analyze_all_volatility(sample_data)
        
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
            assert 'rolling_volatility' in currency_result
            assert 'garch_volatility' in currency_result
            assert 'historical_volatility' in currency_result
            assert 'volatility_regime' in currency_result
            assert 'volatility_metrics' in currency_result
            assert 'volatility_forecast' in currency_result
        
        # Check overall market
        assert 'volatility_metrics' in result['overall_market']
    
    def test_volatility_comparison(self, analyzer, sample_data):
        """Test volatility comparison between currencies"""
        usd_metrics = analyzer.calculate_volatility_metrics(sample_data, currency_code='USD')
        eur_metrics = analyzer.calculate_volatility_metrics(sample_data, currency_code='EUR')
        
        # USD should have higher volatility than EUR based on our test data
        assert usd_metrics['annualized_volatility'] > eur_metrics['annualized_volatility']
    
    def test_empty_dataframe(self, analyzer):
        """Test behavior with empty DataFrame"""
        empty_df = pd.DataFrame(columns=['day', 'currency_code', 'value'])
        
        result = analyzer.calculate_rolling_volatility(empty_df)
        assert len(result['volatility']) == 0
        assert 'message' in result
    
    def test_missing_currency_code(self, analyzer, sample_data):
        """Test behavior when currency code doesn't exist"""
        result = analyzer.calculate_rolling_volatility(sample_data, currency_code='NONEXISTENT')
        assert result['currency_code'] == 'NONEXISTENT'
        assert len(result['volatility']) == 0
        assert 'message' in result
    
    def test_volatility_regime_changes(self, analyzer):
        """Test volatility regime change detection"""
        # Create data with clear regime changes
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        
        # First 30 days: low volatility
        low_vol_data = [{
            'day': date,
            'currency_code': 'USD',
            'value': 100 + i + np.random.normal(0, 0.1)
        } for i, date in enumerate(dates[:30])]
        
        # Next 30 days: high volatility
        high_vol_data = [{
            'day': date,
            'currency_code': 'USD',
            'value': 103 + i + np.random.normal(0, 2)
        } for i, date in enumerate(dates[30:])]
        
        regime_data = pd.DataFrame(low_vol_data + high_vol_data)
        
        result = analyzer.calculate_volatility_regime(regime_data, currency_code='USD')
        
        assert len(result['regimes']) >= 1  # Should detect at least one regime change
        assert result['high_vol_periods'] >= 0
        assert result['low_vol_periods'] >= 0
