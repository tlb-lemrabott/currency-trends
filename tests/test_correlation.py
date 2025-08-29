"""
Unit tests for Correlation Analysis Module

Tests for the CorrelationAnalyzer class and its methods.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analysis.correlation import CorrelationAnalyzer


class TestCorrelationAnalyzer:
    """Test cases for CorrelationAnalyzer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample currency data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Create sample data with known correlation patterns
        usd_data = []
        eur_data = []
        gbp_data = []
        
        for i, date in enumerate(dates):
            # USD: base trend
            usd_value = 100 + (i * 0.1) + np.random.normal(0, 1)
            usd_data.append({
                'day': date,
                'currency_code': 'USD',
                'value': usd_value
            })
            
            # EUR: highly correlated with USD
            eur_value = 120 + (i * 0.08) + np.random.normal(0, 0.8) + (usd_value - 100) * 0.5
            eur_data.append({
                'day': date,
                'currency_code': 'EUR',
                'value': eur_value
            })
            
            # GBP: moderately correlated with USD
            gbp_value = 130 + (i * 0.05) + np.random.normal(0, 1.5) + (usd_value - 100) * 0.3
            gbp_data.append({
                'day': date,
                'currency_code': 'GBP',
                'value': gbp_value
            })
        
        return pd.DataFrame(usd_data + eur_data + gbp_data)
    
    @pytest.fixture
    def analyzer(self):
        """Create CorrelationAnalyzer instance"""
        return CorrelationAnalyzer()
    
    def test_calculate_correlation_matrix_pearson(self, analyzer, sample_data):
        """Test correlation matrix calculation with Pearson method"""
        result = analyzer.calculate_correlation_matrix(sample_data, method='pearson')
        
        assert result['method'] == 'pearson'
        assert 'correlation_matrix' in result
        assert 'currencies' in result
        assert 'data_points' in result
        assert 'date_range' in result
        
        assert len(result['currencies']) == 3  # USD, EUR, GBP
        assert result['data_points'] == 50
        assert 'start' in result['date_range']
        assert 'end' in result['date_range']
        
        # Check correlation matrix structure
        corr_matrix = result['correlation_matrix']
        assert 'USD' in corr_matrix
        assert 'EUR' in corr_matrix
        assert 'GBP' in corr_matrix
    
    def test_calculate_correlation_matrix_spearman(self, analyzer, sample_data):
        """Test correlation matrix calculation with Spearman method"""
        result = analyzer.calculate_correlation_matrix(sample_data, method='spearman')
        
        assert result['method'] == 'spearman'
        assert 'correlation_matrix' in result
        assert len(result['currencies']) == 3
    
    def test_calculate_correlation_matrix_insufficient_currencies(self, analyzer):
        """Test correlation matrix with insufficient currencies"""
        # Create data with only one currency
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        single_currency_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.calculate_correlation_matrix(single_currency_data)
        
        assert 'message' in result
        assert 'Insufficient currencies' in result['message']
    
    def test_calculate_pairwise_correlation(self, analyzer, sample_data):
        """Test pairwise correlation calculation"""
        result = analyzer.calculate_pairwise_correlation(sample_data, 'USD', 'EUR', method='pearson')
        
        assert result['currency1'] == 'USD'
        assert result['currency2'] == 'EUR'
        assert result['method'] == 'pearson'
        assert 'correlation' in result
        assert 'p_value' in result
        assert 'data_points' in result
        assert 'significance' in result
        assert 'strength' in result
        
        assert result['correlation'] is not None
        assert result['p_value'] is not None
        assert result['data_points'] > 0
        assert result['significance'] in ['significant', 'not significant']
        assert result['strength'] in ['very strong', 'strong', 'moderate', 'weak', 'very weak']
    
    def test_calculate_pairwise_correlation_spearman(self, analyzer, sample_data):
        """Test pairwise correlation with Spearman method"""
        result = analyzer.calculate_pairwise_correlation(sample_data, 'USD', 'EUR', method='spearman')
        
        assert result['method'] == 'spearman'
        assert result['correlation'] is not None
        assert result['p_value'] is not None
    
    def test_calculate_pairwise_correlation_nonexistent_currencies(self, analyzer, sample_data):
        """Test pairwise correlation with nonexistent currencies"""
        result = analyzer.calculate_pairwise_correlation(sample_data, 'USD', 'NONEXISTENT', method='pearson')
        
        assert result['currency1'] == 'USD'
        assert result['currency2'] == 'NONEXISTENT'
        assert result['correlation'] is None
        assert result['p_value'] is None
        assert 'message' in result
    
    def test_calculate_rolling_correlation(self, analyzer, sample_data):
        """Test rolling correlation calculation"""
        result = analyzer.calculate_rolling_correlation(sample_data, 'USD', 'EUR', window=20)
        
        assert result['currency1'] == 'USD'
        assert result['currency2'] == 'EUR'
        assert result['method'] == 'pearson'
        assert result['window'] == 20
        assert 'rolling_correlation' in result
        assert 'dates' in result
        assert 'data_points' in result
        
        assert len(result['rolling_correlation']) == 50
        assert len(result['dates']) == 50
        assert result['data_points'] == 50
    
    def test_calculate_rolling_correlation_custom_window(self, analyzer, sample_data):
        """Test rolling correlation with custom window"""
        result = analyzer.calculate_rolling_correlation(sample_data, 'USD', 'EUR', window=10)
        
        assert result['window'] == 10
        assert len(result['rolling_correlation']) == 50
    
    def test_calculate_correlation_clusters(self, analyzer, sample_data):
        """Test correlation cluster identification"""
        result = analyzer.calculate_correlation_clusters(sample_data, method='pearson', threshold=0.5)
        
        assert result['method'] == 'pearson'
        assert result['threshold'] == 0.5
        assert 'clusters' in result
        assert 'high_correlation_pairs' in result
        assert 'total_currencies' in result
        assert 'clustered_currencies' in result
        assert 'unclustered_currencies' in result
        
        assert result['total_currencies'] == 3
        assert result['clustered_currencies'] >= 0
        assert result['unclustered_currencies'] >= 0
    
    def test_calculate_correlation_clusters_high_threshold(self, analyzer, sample_data):
        """Test correlation clusters with high threshold"""
        result = analyzer.calculate_correlation_clusters(sample_data, method='pearson', threshold=0.9)
        
        assert result['threshold'] == 0.9
        # With high threshold, we might have fewer clusters
        assert len(result['clusters']) >= 0
    
    def test_calculate_correlation_stability(self, analyzer, sample_data):
        """Test correlation stability analysis"""
        result = analyzer.calculate_correlation_stability(sample_data, 'USD', 'EUR', window=20)
        
        assert result['currency1'] == 'USD'
        assert result['currency2'] == 'EUR'
        assert result['method'] == 'pearson'
        assert result['window'] == 20
        assert 'mean_correlation' in result
        assert 'std_correlation' in result
        assert 'min_correlation' in result
        assert 'max_correlation' in result
        assert 'correlation_range' in result
        assert 'stability' in result
        assert 'data_points' in result
        
        assert result['stability'] in ['high', 'medium', 'low']
        assert result['data_points'] > 0
    
    def test_calculate_correlation_stability_insufficient_data(self, analyzer):
        """Test correlation stability with insufficient data"""
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        small_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD' if i % 2 == 0 else 'EUR',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.calculate_correlation_stability(small_data, 'USD', 'EUR', window=10)
        
        assert 'message' in result
        assert 'Insufficient data' in result['message']
    
    def test_calculate_correlation_breakdown(self, analyzer, sample_data):
        """Test correlation breakdown analysis"""
        result = analyzer.calculate_correlation_breakdown(sample_data, method='pearson')
        
        assert result['method'] == 'pearson'
        assert 'breakdown' in result
        assert 'total_periods' in result
        assert 'date_range' in result
        
        breakdown = result['breakdown']
        assert 'yearly' in breakdown
        assert 'start' in result['date_range']
        assert 'end' in result['date_range']
    
    def test_calculate_correlation_breakdown_insufficient_currencies(self, analyzer):
        """Test correlation breakdown with insufficient currencies"""
        # Create data with only one currency
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        single_currency_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.calculate_correlation_breakdown(single_currency_data)
        
        assert 'message' in result
        assert 'Insufficient currencies' in result['message']
    
    def test_classify_correlation_strength(self, analyzer):
        """Test correlation strength classification"""
        # Test different correlation values
        assert analyzer._classify_correlation_strength(0.9) == 'very strong'
        assert analyzer._classify_correlation_strength(0.7) == 'strong'
        assert analyzer._classify_correlation_strength(0.5) == 'moderate'
        assert analyzer._classify_correlation_strength(0.3) == 'weak'
        assert analyzer._classify_correlation_strength(0.1) == 'very weak'
        
        # Test negative correlations
        assert analyzer._classify_correlation_strength(-0.9) == 'very strong'
        assert analyzer._classify_correlation_strength(-0.7) == 'strong'
    
    def test_analyze_all_correlations(self, analyzer, sample_data):
        """Test comprehensive correlation analysis"""
        result = analyzer.analyze_all_correlations(sample_data)
        
        assert 'summary' in result
        assert 'pearson_correlation_matrix' in result
        assert 'spearman_correlation_matrix' in result
        assert 'correlation_clusters' in result
        assert 'correlation_breakdown' in result
        assert 'pairwise_correlations' in result
        assert 'correlation_stability' in result
        
        summary = result['summary']
        assert summary['total_currencies'] == 3
        assert 'analysis_date' in summary
        
        # Check pairwise correlations
        pairwise = result['pairwise_correlations']
        assert len(pairwise) == 3  # 3 pairs: USD-EUR, USD-GBP, EUR-GBP
    
    def test_analyze_all_correlations_insufficient_currencies(self, analyzer):
        """Test comprehensive analysis with insufficient currencies"""
        # Create data with only one currency
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        single_currency_data = pd.DataFrame([{
            'day': date,
            'currency_code': 'USD',
            'value': 100.0 + i
        } for i, date in enumerate(dates)])
        
        result = analyzer.analyze_all_correlations(single_currency_data)
        
        assert 'summary' in result
        assert 'message' in result['summary']
        assert 'Insufficient currencies' in result['summary']['message']
    
    def test_correlation_patterns(self, analyzer, sample_data):
        """Test that correlation patterns are detected correctly"""
        # EUR should be more correlated with USD than GBP
        usd_eur = analyzer.calculate_pairwise_correlation(sample_data, 'USD', 'EUR')
        usd_gbp = analyzer.calculate_pairwise_correlation(sample_data, 'USD', 'GBP')
        
        assert usd_eur['correlation'] is not None
        assert usd_gbp['correlation'] is not None
        
        # EUR should have higher correlation with USD than GBP (based on our test data)
        assert abs(usd_eur['correlation']) >= abs(usd_gbp['correlation'])
    
    def test_empty_dataframe(self, analyzer):
        """Test behavior with empty DataFrame"""
        empty_df = pd.DataFrame(columns=['day', 'currency_code', 'value'])
        
        result = analyzer.calculate_correlation_matrix(empty_df)
        assert 'message' in result
        assert 'Insufficient currencies' in result['message']
    
    def test_missing_data_handling(self, analyzer, sample_data):
        """Test handling of missing data"""
        # Add some missing values
        sample_data_with_nulls = sample_data.copy()
        sample_data_with_nulls.loc[0, 'value'] = np.nan
        
        result = analyzer.calculate_pairwise_correlation(sample_data_with_nulls, 'USD', 'EUR')
        
        # Should still work, just with fewer data points
        assert result['correlation'] is not None
        assert result['data_points'] < len(sample_data) / 2  # Less than half due to nulls
    
    def test_invalid_correlation_method(self, analyzer, sample_data):
        """Test handling of invalid correlation method"""
        with pytest.raises(ValueError):
            analyzer.calculate_correlation_matrix(sample_data, method='invalid_method')
        
        with pytest.raises(ValueError):
            analyzer.calculate_pairwise_correlation(sample_data, 'USD', 'EUR', method='invalid_method')
