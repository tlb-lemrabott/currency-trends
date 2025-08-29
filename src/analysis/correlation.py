"""
Correlation Analysis Module

This module handles correlation calculation and analysis for currency exchange rate data.
Implements single responsibility principle by focusing only on correlation analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyzer for currency exchange rate correlations"""
    
    def __init__(self):
        """Initialize the correlation analyzer"""
        self.correlation_results = {}
    
    def calculate_correlation_matrix(self, df: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
        """
        Calculate correlation matrix for all currencies
        
        Args:
            df: DataFrame with currency data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dict: Correlation matrix results
        """
        if len(df['currency_code'].unique()) < 2:
            return {
                'method': method,
                'correlation_matrix': {},
                'message': 'Insufficient currencies for correlation analysis (minimum 2 required)'
            }
        
        # Pivot data to get currencies as columns
        pivot_df = df.pivot(index='day', columns='currency_code', values='value')
        
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = pivot_df.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = pivot_df.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = pivot_df.corr(method='kendalltau')
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        
        result = {
            'method': method,
            'correlation_matrix': corr_matrix.to_dict(),
            'currencies': list(corr_matrix.columns),
            'data_points': len(pivot_df),
            'date_range': {
                'start': pivot_df.index.min().strftime('%Y-%m-%d'),
                'end': pivot_df.index.max().strftime('%Y-%m-%d')
            }
        }
        
        return result
    
    def calculate_pairwise_correlation(self, df: pd.DataFrame, currency1: str, currency2: str,
                                    method: str = 'pearson') -> Dict[str, Any]:
        """
        Calculate correlation between two specific currencies
        
        Args:
            df: DataFrame with currency data
            currency1: First currency code
            currency2: Second currency code
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dict: Pairwise correlation results
        """
        # Filter data for the two currencies
        df_filtered = df[df['currency_code'].isin([currency1, currency2])].copy()
        
        if len(df_filtered) == 0:
            return {
                'currency1': currency1,
                'currency2': currency2,
                'method': method,
                'correlation': None,
                'p_value': None,
                'message': 'No data available for specified currencies'
            }
        
        # Pivot data
        pivot_df = df_filtered.pivot(index='day', columns='currency_code', values='value')
        
        if currency1 not in pivot_df.columns or currency2 not in pivot_df.columns:
            return {
                'currency1': currency1,
                'currency2': currency2,
                'method': method,
                'correlation': None,
                'p_value': None,
                'message': 'One or both currencies not found in data'
            }
        
        # Remove rows with missing values
        pivot_df = pivot_df.dropna()
        
        if len(pivot_df) < 2:
            return {
                'currency1': currency1,
                'currency2': currency2,
                'method': method,
                'correlation': None,
                'p_value': None,
                'message': 'Insufficient overlapping data points'
            }
        
        # Calculate correlation
        x = pivot_df[currency1].values
        y = pivot_df[currency2].values
        
        if method == 'pearson':
            corr, p_value = pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = spearmanr(x, y)
        elif method == 'kendall':
            corr, p_value = kendalltau(x, y)
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        
        result = {
            'currency1': currency1,
            'currency2': currency2,
            'method': method,
            'correlation': corr,
            'p_value': p_value,
            'data_points': len(pivot_df),
            'significance': 'significant' if p_value < 0.05 else 'not significant',
            'strength': self._classify_correlation_strength(corr)
        }
        
        return result
    
    def calculate_rolling_correlation(self, df: pd.DataFrame, currency1: str, currency2: str,
                                   window: int = 30, method: str = 'pearson') -> Dict[str, Any]:
        """
        Calculate rolling correlation between two currencies
        
        Args:
            df: DataFrame with currency data
            currency1: First currency code
            currency2: Second currency code
            window: Rolling window size
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dict: Rolling correlation results
        """
        # Filter data for the two currencies
        df_filtered = df[df['currency_code'].isin([currency1, currency2])].copy()
        
        if len(df_filtered) == 0:
            return {
                'currency1': currency1,
                'currency2': currency2,
                'method': method,
                'window': window,
                'rolling_correlation': [],
                'dates': [],
                'message': 'No data available for specified currencies'
            }
        
        # Pivot data
        pivot_df = df_filtered.pivot(index='day', columns='currency_code', values='value')
        
        if currency1 not in pivot_df.columns or currency2 not in pivot_df.columns:
            return {
                'currency1': currency1,
                'currency2': currency2,
                'method': method,
                'window': window,
                'rolling_correlation': [],
                'dates': [],
                'message': 'One or both currencies not found in data'
            }
        
        # Calculate rolling correlation
        if method == 'pearson':
            rolling_corr = pivot_df[currency1].rolling(window=window).corr(pivot_df[currency2])
        elif method == 'spearman':
            # For spearman, we need to calculate manually
            rolling_corr = pivot_df[currency1].rolling(window=window).corr(pivot_df[currency2], method='spearman')
        elif method == 'kendall':
            # For kendall, we need to calculate manually
            rolling_corr = pivot_df[currency1].rolling(window=window).corr(pivot_df[currency2], method='kendalltau')
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        
        result = {
            'currency1': currency1,
            'currency2': currency2,
            'method': method,
            'window': window,
            'rolling_correlation': rolling_corr.tolist(),
            'dates': pivot_df.index.strftime('%Y-%m-%d').tolist(),
            'data_points': len(pivot_df)
        }
        
        return result
    
    def calculate_correlation_clusters(self, df: pd.DataFrame, method: str = 'pearson',
                                    threshold: float = 0.7) -> Dict[str, Any]:
        """
        Identify clusters of highly correlated currencies
        
        Args:
            df: DataFrame with currency data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            threshold: Correlation threshold for clustering
            
        Returns:
            Dict: Correlation cluster results
        """
        # Calculate correlation matrix
        corr_result = self.calculate_correlation_matrix(df, method)
        
        if 'message' in corr_result:
            return {
                'method': method,
                'threshold': threshold,
                'clusters': [],
                'message': corr_result['message']
            }
        
        corr_matrix = pd.DataFrame(corr_result['correlation_matrix'])
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append({
                        'currency1': corr_matrix.columns[i],
                        'currency2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': self._classify_correlation_strength(corr_value)
                    })
        
        # Sort by absolute correlation value
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Group into clusters
        clusters = []
        used_currencies = set()
        
        for pair in high_corr_pairs:
            curr1, curr2 = pair['currency1'], pair['currency2']
            
            # Check if either currency is already in a cluster
            found_cluster = False
            for cluster in clusters:
                if curr1 in cluster['currencies'] or curr2 in cluster['currencies']:
                    # Add both currencies to the cluster
                    cluster['currencies'].add(curr1)
                    cluster['currencies'].add(curr2)
                    cluster['pairs'].append(pair)
                    found_cluster = True
                    break
            
            if not found_cluster:
                # Create new cluster
                clusters.append({
                    'currencies': {curr1, curr2},
                    'pairs': [pair],
                    'avg_correlation': pair['correlation']
                })
            
            used_currencies.add(curr1)
            used_currencies.add(curr2)
        
        # Convert sets to lists for JSON serialization
        for cluster in clusters:
            cluster['currencies'] = list(cluster['currencies'])
            cluster['avg_correlation'] = sum(p['correlation'] for p in cluster['pairs']) / len(cluster['pairs'])
        
        result = {
            'method': method,
            'threshold': threshold,
            'clusters': clusters,
            'high_correlation_pairs': high_corr_pairs,
            'total_currencies': len(corr_matrix.columns),
            'clustered_currencies': len(used_currencies),
            'unclustered_currencies': len(corr_matrix.columns) - len(used_currencies)
        }
        
        return result
    
    def calculate_correlation_stability(self, df: pd.DataFrame, currency1: str, currency2: str,
                                     window: int = 30, method: str = 'pearson') -> Dict[str, Any]:
        """
        Analyze correlation stability over time
        
        Args:
            df: DataFrame with currency data
            currency1: First currency code
            currency2: Second currency code
            window: Rolling window size
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dict: Correlation stability analysis results
        """
        # Get rolling correlation
        rolling_result = self.calculate_rolling_correlation(df, currency1, currency2, window, method)
        
        if 'message' in rolling_result:
            return {
                'currency1': currency1,
                'currency2': currency2,
                'method': method,
                'window': window,
                'message': rolling_result['message']
            }
        
        rolling_corr = rolling_result['rolling_correlation']
        valid_corr = [c for c in rolling_corr if not pd.isna(c)]
        
        if len(valid_corr) < 2:
            return {
                'currency1': currency1,
                'currency2': currency2,
                'method': method,
                'window': window,
                'message': 'Insufficient data for stability analysis'
            }
        
        # Calculate stability metrics
        mean_corr = np.mean(valid_corr)
        std_corr = np.std(valid_corr)
        min_corr = np.min(valid_corr)
        max_corr = np.max(valid_corr)
        
        # Calculate correlation of correlations (meta-correlation)
        if len(valid_corr) > 10:
            time_index = np.arange(len(valid_corr))
            meta_corr, meta_p_value = pearsonr(time_index, valid_corr)
        else:
            meta_corr, meta_p_value = None, None
        
        # Classify stability
        if std_corr < 0.1:
            stability = 'high'
        elif std_corr < 0.2:
            stability = 'medium'
        else:
            stability = 'low'
        
        result = {
            'currency1': currency1,
            'currency2': currency2,
            'method': method,
            'window': window,
            'mean_correlation': mean_corr,
            'std_correlation': std_corr,
            'min_correlation': min_corr,
            'max_correlation': max_corr,
            'correlation_range': max_corr - min_corr,
            'stability': stability,
            'meta_correlation': meta_corr,
            'meta_p_value': meta_p_value,
            'data_points': len(valid_corr)
        }
        
        return result
    
    def calculate_correlation_breakdown(self, df: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
        """
        Analyze correlation breakdown by time periods
        
        Args:
            df: DataFrame with currency data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dict: Correlation breakdown analysis results
        """
        if len(df['currency_code'].unique()) < 2:
            return {
                'method': method,
                'breakdown': {},
                'message': 'Insufficient currencies for correlation analysis'
            }
        
        # Add time-based features
        df_copy = df.copy()
        df_copy['year'] = df_copy['day'].dt.year
        df_copy['month'] = df_copy['day'].dt.month
        df_copy['quarter'] = df_copy['day'].dt.quarter
        
        breakdown = {}
        
        # Yearly breakdown
        yearly_corr = {}
        for year in df_copy['year'].unique():
            year_data = df_copy[df_copy['year'] == year]
            if len(year_data['currency_code'].unique()) >= 2:
                year_result = self.calculate_correlation_matrix(year_data, method)
                if 'correlation_matrix' in year_result:
                    yearly_corr[str(year)] = year_result['correlation_matrix']
        
        breakdown['yearly'] = yearly_corr
        
        # Monthly breakdown (if we have enough data)
        if len(df_copy) > 60:  # At least 2 months of data
            monthly_corr = {}
            for year in df_copy['year'].unique():
                for month in df_copy[df_copy['year'] == year]['month'].unique():
                    month_data = df_copy[(df_copy['year'] == year) & (df_copy['month'] == month)]
                    if len(month_data['currency_code'].unique()) >= 2:
                        month_result = self.calculate_correlation_matrix(month_data, method)
                        if 'correlation_matrix' in month_result:
                            key = f"{year}-{month:02d}"
                            monthly_corr[key] = month_result['correlation_matrix']
            
            breakdown['monthly'] = monthly_corr
        
        result = {
            'method': method,
            'breakdown': breakdown,
            'total_periods': len(breakdown.get('yearly', {})) + len(breakdown.get('monthly', {})),
            'date_range': {
                'start': df_copy['day'].min().strftime('%Y-%m-%d'),
                'end': df_copy['day'].max().strftime('%Y-%m-%d')
            }
        }
        
        return result
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """
        Classify correlation strength based on absolute value
        
        Args:
            correlation: Correlation coefficient
            
        Returns:
            str: Strength classification
        """
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.8:
            return 'very strong'
        elif abs_corr >= 0.6:
            return 'strong'
        elif abs_corr >= 0.4:
            return 'moderate'
        elif abs_corr >= 0.2:
            return 'weak'
        else:
            return 'very weak'
    
    def analyze_all_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis for all currencies
        
        Args:
            df: DataFrame with currency data
            
        Returns:
            Dict: Comprehensive correlation analysis results
        """
        currencies = df['currency_code'].unique()
        
        if len(currencies) < 2:
            return {
                'summary': {
                    'total_currencies': len(currencies),
                    'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'message': 'Insufficient currencies for correlation analysis'
                }
            }
        
        results = {
            'summary': {
                'total_currencies': len(currencies),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Calculate correlation matrix for different methods
        for method in ['pearson', 'spearman']:
            results[f'{method}_correlation_matrix'] = self.calculate_correlation_matrix(df, method)
        
        # Calculate correlation clusters
        results['correlation_clusters'] = self.calculate_correlation_clusters(df, method='pearson')
        
        # Calculate correlation breakdown
        results['correlation_breakdown'] = self.calculate_correlation_breakdown(df, method='pearson')
        
        # Calculate pairwise correlations for all currency pairs
        pairwise_correlations = []
        for i, curr1 in enumerate(currencies):
            for j, curr2 in enumerate(currencies):
                if i < j:  # Avoid duplicate pairs
                    pair_result = self.calculate_pairwise_correlation(df, curr1, curr2, method='pearson')
                    if 'correlation' in pair_result and pair_result['correlation'] is not None:
                        pairwise_correlations.append(pair_result)
        
        results['pairwise_correlations'] = pairwise_correlations
        
        # Calculate correlation stability for all pairs
        stability_analysis = []
        for pair in pairwise_correlations:
            stability = self.calculate_correlation_stability(
                df, pair['currency1'], pair['currency2'], method='pearson'
            )
            if 'stability' in stability:
                stability_analysis.append(stability)
        
        results['correlation_stability'] = stability_analysis
        
        return results
