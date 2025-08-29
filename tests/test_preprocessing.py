"""
Unit tests for data preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.preprocessing import DataPreprocessor
from src.data.schema import CurrencyData, Currency, ExchangeRate, DataValidator


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""

    @pytest.fixture
    def preprocessor(self):
        """Create data preprocessor instance"""
        return DataPreprocessor()

    @pytest.fixture
    def sample_currency_data(self):
        """Create sample currency data for testing"""
        exchange_rates = [
            ExchangeRate(
                id=137058,
                day="2016-06-14",
                value="333.21",
                end_date="2016-06-15"
            ),
            ExchangeRate(
                id=137059,
                day="2016-06-15",
                value="334.50",
                end_date="2016-06-16"
            ),
            ExchangeRate(
                id=137060,
                day="2016-06-16",
                value="335.00",
                end_date="2016-06-17"
            )
        ]
        
        currency = Currency(
            id=1,
            name_fr="Dollar US",
            name_ar="الدولار الأمريكي",
            unity=1,
            code="USD",
            exchange_rates=exchange_rates
        )
        
        return CurrencyData(
            success=True,
            message="Success",
            data=[currency]
        )

    def test_convert_to_dataframe(self, preprocessor, sample_currency_data):
        """Test converting currency data to DataFrame"""
        df = preprocessor.convert_to_dataframe(sample_currency_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == [
            'currency_id', 'currency_code', 'currency_name_fr', 'currency_name_ar',
            'unity', 'rate_id', 'day', 'value', 'end_date'
        ]
        assert df['currency_code'].iloc[0] == "USD"
        assert df['value'].iloc[0] == 333.21

    def test_handle_missing_values_interpolate(self, preprocessor):
        """Test handling missing values with interpolation"""
        # Create DataFrame with missing values
        df = pd.DataFrame({
            'currency_code': ['USD', 'USD', 'USD', 'USD'],
            'day': pd.date_range('2016-06-14', periods=4),
            'value': [333.21, np.nan, 335.00, 336.50]
        })
        
        result = preprocessor.handle_missing_values(df, method='interpolate')
        
        assert not result['value'].isnull().any()
        assert len(result) == 4

    def test_handle_missing_values_forward_fill(self, preprocessor):
        """Test handling missing values with forward fill"""
        df = pd.DataFrame({
            'currency_code': ['USD', 'USD', 'USD', 'USD'],
            'day': pd.date_range('2016-06-14', periods=4),
            'value': [333.21, np.nan, 335.00, 336.50]
        })
        
        result = preprocessor.handle_missing_values(df, method='forward_fill')
        
        assert not result['value'].isnull().any()
        assert result['value'].iloc[1] == 333.21  # Should be filled with previous value

    def test_handle_missing_values_drop(self, preprocessor):
        """Test handling missing values by dropping"""
        df = pd.DataFrame({
            'currency_code': ['USD', 'USD', 'USD', 'USD'],
            'day': pd.date_range('2016-06-14', periods=4),
            'value': [333.21, np.nan, 335.00, 336.50]
        })
        
        result = preprocessor.handle_missing_values(df, method='drop')
        
        assert len(result) == 3  # One row should be dropped
        assert not result['value'].isnull().any()

    def test_detect_and_handle_outliers_iqr(self, preprocessor):
        """Test outlier detection and handling with IQR method"""
        # Create DataFrame with outliers
        df = pd.DataFrame({
            'currency_code': ['USD'] * 10,
            'day': pd.date_range('2016-06-14', periods=10),
            'value': [333.21, 334.50, 335.00, 336.50, 337.00, 338.00, 339.00, 340.00, 500.00, 341.00]  # 500 is outlier
        })
        
        result = preprocessor.detect_and_handle_outliers(df, method='iqr')
        
        assert len(result) == 10
        assert result['value'].max() < 500.00  # Outlier should be handled

    def test_detect_and_handle_outliers_zscore(self, preprocessor):
        """Test outlier detection and handling with Z-score method"""
        df = pd.DataFrame({
            'currency_code': ['USD'] * 10,
            'day': pd.date_range('2016-06-14', periods=10),
            'value': [333.21, 334.50, 335.00, 336.50, 337.00, 338.00, 339.00, 340.00, 500.00, 341.00]
        })
        
        result = preprocessor.detect_and_handle_outliers(df, method='zscore', threshold=2.0)
        
        assert len(result) == 10
        assert result['value'].max() < 500.00

    def test_normalize_values_minmax(self, preprocessor):
        """Test value normalization with min-max method"""
        df = pd.DataFrame({
            'currency_code': ['USD'] * 5,
            'day': pd.date_range('2016-06-14', periods=5),
            'value': [333.21, 334.50, 335.00, 336.50, 337.00]
        })
        
        result = preprocessor.normalize_values(df, method='minmax')
        
        assert 'value_normalized' in result.columns
        assert result['value_normalized'].min() == 0.0
        assert result['value_normalized'].max() == 1.0

    def test_normalize_values_zscore(self, preprocessor):
        """Test value normalization with Z-score method"""
        df = pd.DataFrame({
            'currency_code': ['USD'] * 5,
            'day': pd.date_range('2016-06-14', periods=5),
            'value': [333.21, 334.50, 335.00, 336.50, 337.00]
        })
        
        result = preprocessor.normalize_values(df, method='zscore')
        
        assert 'value_normalized' in result.columns
        assert abs(result['value_normalized'].mean()) < 1e-10  # Should be close to 0
        assert abs(result['value_normalized'].std() - 1.0) < 1e-10  # Should be close to 1

    def test_add_technical_indicators(self, preprocessor):
        """Test adding technical indicators"""
        df = pd.DataFrame({
            'currency_code': ['USD'] * 30,
            'day': pd.date_range('2016-06-14', periods=30),
            'value': np.random.uniform(330, 340, 30)
        })
        
        result = preprocessor.add_technical_indicators(df)
        
        # Check that technical indicators were added
        expected_indicators = ['sma_7', 'sma_30', 'ema_12', 'volatility_7', 'volatility_30', 
                             'price_change', 'price_change_pct', 'bb_upper', 'bb_lower', 'bb_position']
        
        for indicator in expected_indicators:
            assert indicator in result.columns

    def test_add_time_features(self, preprocessor):
        """Test adding time features"""
        df = pd.DataFrame({
            'currency_code': ['USD'] * 10,
            'day': pd.date_range('2016-06-14', periods=10),
            'value': [333.21, 334.50, 335.00, 336.50, 337.00, 338.00, 339.00, 340.00, 341.00, 342.00]
        })
        
        result = preprocessor.add_time_features(df)
        
        # Check that time features were added
        expected_features = ['year', 'month', 'day_of_week', 'day_of_year', 'quarter',
                           'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
        
        for feature in expected_features:
            assert feature in result.columns

    def test_resample_data_daily_to_weekly(self, preprocessor):
        """Test resampling data from daily to weekly"""
        df = pd.DataFrame({
            'currency_code': ['USD'] * 14,
            'day': pd.date_range('2016-06-14', periods=14),
            'value': [333.21, 334.50, 335.00, 336.50, 337.00, 338.00, 339.00, 
                     340.00, 341.00, 342.00, 343.00, 344.00, 345.00, 346.00]
        })
        
        result = preprocessor.resample_data(df, frequency='W', method='last')
        
        assert len(result) < len(df)  # Should have fewer rows after resampling
        assert 'currency_code' in result.columns

    def test_create_lagged_features(self, preprocessor):
        """Test creating lagged features"""
        df = pd.DataFrame({
            'currency_code': ['USD'] * 10,
            'day': pd.date_range('2016-06-14', periods=10),
            'value': [333.21, 334.50, 335.00, 336.50, 337.00, 338.00, 339.00, 340.00, 341.00, 342.00],
            'price_change': [0, 1.29, 0.50, 1.50, 0.50, 1.00, 1.00, 1.00, 1.00, 1.00]
        })
        
        result = preprocessor.create_lagged_features(df, lags=[1, 2])
        
        # Check that lagged features were added
        assert 'value_lag_1' in result.columns
        assert 'value_lag_2' in result.columns
        assert 'price_change_lag_1' in result.columns
        assert 'price_change_lag_2' in result.columns

    def test_get_preprocessing_summary(self, preprocessor):
        """Test generating preprocessing summary"""
        original_df = pd.DataFrame({
            'currency_code': ['USD'] * 5,
            'day': pd.date_range('2016-06-14', periods=5),
            'value': [333.21, 334.50, 335.00, 336.50, 337.00]
        })
        
        processed_df = original_df.copy()
        processed_df['new_feature'] = [1, 2, 3, 4, 5]
        
        summary = preprocessor.get_preprocessing_summary(original_df, processed_df)
        
        assert 'original_shape' in summary
        assert 'processed_shape' in summary
        assert 'currencies_processed' in summary
        assert 'date_range' in summary
        assert 'features_added' in summary
        assert 'new_feature' in summary['features_added']

    def test_preprocess_complete(self, preprocessor, sample_currency_data):
        """Test complete preprocessing pipeline"""
        config = {
            'handle_missing': 'interpolate',
            'handle_outliers': 'iqr',
            'outlier_threshold': 1.5,
            'normalize': 'minmax',
            'add_indicators': True,
            'add_time_features': True,
            'lags': [1, 2]
        }
        
        df, summary = preprocessor.preprocess_complete(sample_currency_data, config)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(summary, dict)
        assert len(df) > 0
        assert 'value_normalized' in df.columns
        assert 'sma_7' in df.columns
        assert 'year' in df.columns

    def test_preprocess_complete_default_config(self, preprocessor, sample_currency_data):
        """Test complete preprocessing pipeline with default config"""
        df, summary = preprocessor.preprocess_complete(sample_currency_data)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(summary, dict)
        assert len(df) > 0

    def test_invalid_methods(self, preprocessor):
        """Test handling of invalid methods"""
        df = pd.DataFrame({
            'currency_code': ['USD'] * 5,
            'day': pd.date_range('2016-06-14', periods=5),
            'value': [333.21, 334.50, 335.00, 336.50, 337.00]
        })
        
        # Test invalid missing value method
        with pytest.raises(ValueError, match="Unknown method"):
            preprocessor.handle_missing_values(df, method='invalid_method')
        
        # Test invalid outlier detection method
        with pytest.raises(ValueError, match="Unknown method"):
            preprocessor.detect_and_handle_outliers(df, method='invalid_method')
        
        # Test invalid normalization method
        with pytest.raises(ValueError, match="Unknown method"):
            preprocessor.normalize_values(df, method='invalid_method')
        
        # Test invalid resampling method
        with pytest.raises(ValueError, match="Unknown method"):
            preprocessor.resample_data(df, method='invalid_method')

    def test_multiple_currencies(self, preprocessor):
        """Test preprocessing with multiple currencies"""
        # Create data with multiple currencies
        df = pd.DataFrame({
            'currency_code': ['USD'] * 5 + ['EUR'] * 5,
            'day': pd.date_range('2016-06-14', periods=10),
            'value': [333.21, 334.50, 335.00, 336.50, 337.00, 423.21, 424.50, 425.00, 426.50, 427.00]
        })
        
        # Test normalization with multiple currencies
        result = preprocessor.normalize_values(df, method='minmax')
        
        assert len(result) == 10
        assert result['currency_code'].nunique() == 2
        
        # Check that each currency is normalized separately
        usd_normalized = result[result['currency_code'] == 'USD']['value_normalized']
        eur_normalized = result[result['currency_code'] == 'EUR']['value_normalized']
        
        assert usd_normalized.min() == 0.0
        assert usd_normalized.max() == 1.0
        assert eur_normalized.min() == 0.0
        assert eur_normalized.max() == 1.0
