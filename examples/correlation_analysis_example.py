"""
Correlation Analysis Example Script

This script demonstrates the correlation analysis functionality with concrete results.
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import DataValidator
from src.data.preprocessing import DataPreprocessor
from src.analysis.correlation import CorrelationAnalyzer


def main():
    """Main function demonstrating correlation analysis"""
    print("=== Currency Correlation Analysis Example ===\n")
    
    # Load and preprocess data
    print("1. Loading and preprocessing data...")
    currency_data = DataValidator.load_and_validate_from_file('data/sample_currency_data.json')
    preprocessor = DataPreprocessor()
    df = preprocessor.convert_to_dataframe(currency_data)
    
    # Add some technical indicators
    df = preprocessor.add_technical_indicators(df)
    df = preprocessor.add_time_features(df)
    
    print(f"   ✓ Loaded {len(currency_data.data)} currencies")
    print(f"   ✓ DataFrame shape: {df.shape}")
    
    # Initialize correlation analyzer
    analyzer = CorrelationAnalyzer()
    
    # Analyze correlations
    print("\n2. Analyzing currency correlations...")
    
    # Correlation matrix
    print("\n--- Correlation Matrix Analysis ---")
    pearson_matrix = analyzer.calculate_correlation_matrix(df, method='pearson')
    if 'correlation_matrix' in pearson_matrix:
        print(f"   Method: {pearson_matrix['method']}")
        print(f"   Currencies: {pearson_matrix['currencies']}")
        print(f"   Data points: {pearson_matrix['data_points']}")
        print(f"   Date range: {pearson_matrix['date_range']['start']} to {pearson_matrix['date_range']['end']}")
        
        # Show correlation values
        corr_matrix = pearson_matrix['correlation_matrix']
        for curr1 in pearson_matrix['currencies']:
            for curr2 in pearson_matrix['currencies']:
                if curr1 != curr2:
                    corr_value = corr_matrix[curr1][curr2]
                    print(f"     {curr1}-{curr2}: {corr_value:.4f}")
    else:
        print(f"   {pearson_matrix['message']}")
    
    # Pairwise correlations
    print("\n--- Pairwise Correlation Analysis ---")
    currencies = df['currency_code'].unique()
    for i, curr1 in enumerate(currencies):
        for j, curr2 in enumerate(currencies):
            if i < j:  # Avoid duplicate pairs
                pair_result = analyzer.calculate_pairwise_correlation(df, curr1, curr2, method='pearson')
                if 'correlation' in pair_result and pair_result['correlation'] is not None:
                    print(f"   {curr1}-{curr2}:")
                    print(f"     - Correlation: {pair_result['correlation']:.4f}")
                    print(f"     - P-value: {pair_result['p_value']:.4f}")
                    print(f"     - Significance: {pair_result['significance']}")
                    print(f"     - Strength: {pair_result['strength']}")
                    print(f"     - Data points: {pair_result['data_points']}")
    
    # Correlation clusters
    print("\n--- Correlation Cluster Analysis ---")
    clusters_result = analyzer.calculate_correlation_clusters(df, method='pearson', threshold=0.5)
    if 'clusters' in clusters_result:
        print(f"   Threshold: {clusters_result['threshold']}")
        print(f"   Total currencies: {clusters_result['total_currencies']}")
        print(f"   Clustered currencies: {clusters_result['clustered_currencies']}")
        print(f"   Unclustered currencies: {clusters_result['unclustered_currencies']}")
        
        if clusters_result['clusters']:
            print(f"   Clusters found: {len(clusters_result['clusters'])}")
            for i, cluster in enumerate(clusters_result['clusters']):
                print(f"     Cluster {i+1}: {cluster['currencies']}")
                print(f"       - Average correlation: {cluster['avg_correlation']:.4f}")
                print(f"       - Pairs: {len(cluster['pairs'])}")
        else:
            print("   No clusters found with current threshold")
    
    # Rolling correlation
    print("\n--- Rolling Correlation Analysis ---")
    if len(currencies) >= 2:
        rolling_result = analyzer.calculate_rolling_correlation(df, currencies[0], currencies[1], window=10)
        if 'rolling_correlation' in rolling_result:
            print(f"   {rolling_result['currency1']}-{rolling_result['currency2']}:")
            print(f"     - Method: {rolling_result['method']}")
            print(f"     - Window: {rolling_result['window']}")
            print(f"     - Data points: {rolling_result['data_points']}")
            
            # Show some correlation values
            valid_corr = [c for c in rolling_result['rolling_correlation'] if not pd.isna(c)]
            if valid_corr:
                print(f"     - Current correlation: {valid_corr[-1]:.4f}")
                print(f"     - Average correlation: {sum(valid_corr)/len(valid_corr):.4f}")
                print(f"     - Min correlation: {min(valid_corr):.4f}")
                print(f"     - Max correlation: {max(valid_corr):.4f}")
    
    # Correlation stability
    print("\n--- Correlation Stability Analysis ---")
    if len(currencies) >= 2:
        stability_result = analyzer.calculate_correlation_stability(df, currencies[0], currencies[1])
        if 'stability' in stability_result:
            print(f"   {stability_result['currency1']}-{stability_result['currency2']}:")
            print(f"     - Stability: {stability_result['stability']}")
            print(f"     - Mean correlation: {stability_result['mean_correlation']:.4f}")
            print(f"     - Std correlation: {stability_result['std_correlation']:.4f}")
            print(f"     - Correlation range: {stability_result['correlation_range']:.4f}")
            print(f"     - Data points: {stability_result['data_points']}")
    
    # Correlation breakdown
    print("\n--- Correlation Breakdown Analysis ---")
    breakdown_result = analyzer.calculate_correlation_breakdown(df, method='pearson')
    if 'breakdown' in breakdown_result:
        print(f"   Method: {breakdown_result['method']}")
        print(f"   Total periods: {breakdown_result['total_periods']}")
        
        breakdown = breakdown_result['breakdown']
        if 'yearly' in breakdown and breakdown['yearly']:
            print(f"   Yearly breakdown available: {len(breakdown['yearly'])} years")
        if 'monthly' in breakdown and breakdown['monthly']:
            print(f"   Monthly breakdown available: {len(breakdown['monthly'])} months")
    
    # Comprehensive analysis
    print("\n3. Running comprehensive correlation analysis...")
    comprehensive_results = analyzer.analyze_all_correlations(df)
    print(f"   ✓ Analyzed {comprehensive_results['summary']['total_currencies']} currencies")
    print(f"   ✓ Analysis completed at: {comprehensive_results['summary']['analysis_date']}")
    
    # Summary statistics
    print("\n4. Correlation Summary Statistics:")
    if 'pairwise_correlations' in comprehensive_results:
        for pair in comprehensive_results['pairwise_correlations']:
            print(f"   {pair['currency1']}-{pair['currency2']}:")
            print(f"     - Correlation: {pair['correlation']:.4f}")
            print(f"     - Strength: {pair['strength']}")
            print(f"     - Significance: {pair['significance']}")
    
    print("\n=== Correlation Analysis Example completed successfully! ===")
    print("Check the dashboard at http://localhost:8050 for interactive visualizations.")


if __name__ == "__main__":
    main()
