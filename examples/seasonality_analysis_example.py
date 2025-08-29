"""
Seasonality Analysis Example Script

This script demonstrates the seasonality analysis functionality with concrete results.
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import DataValidator
from src.data.preprocessing import DataPreprocessor
from src.analysis.seasonality import SeasonalityAnalyzer


def main():
    """Main function demonstrating seasonality analysis"""
    print("=== Currency Seasonality Analysis Example ===\n")
    
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
    
    # Initialize seasonality analyzer
    analyzer = SeasonalityAnalyzer()
    
    # Analyze seasonality for each currency
    print("\n2. Analyzing seasonality for each currency...")
    
    for currency in df['currency_code'].unique():
        print(f"\n--- {currency} Seasonality Analysis ---")
        
        # Seasonal pattern detection
        seasonal_result = analyzer.detect_seasonal_patterns(df, currency_code=currency, period=12)
        if 'seasonal_strength' in seasonal_result:
            print(f"   Seasonal Pattern Detection:")
            print(f"     - Period: {seasonal_result['period']}")
            print(f"     - Seasonal Strength: {seasonal_result['seasonal_strength']:.4f}")
            print(f"     - Data points: {seasonal_result['data_points']}")
            print(f"     - Date range: {seasonal_result['date_range']['start']} to {seasonal_result['date_range']['end']}")
            
            if seasonal_result['seasonal_peaks']:
                print(f"     - Seasonal peaks detected: {len(seasonal_result['seasonal_peaks'])}")
        else:
            print(f"   {seasonal_result['message']}")
        
        # Monthly patterns
        monthly_result = analyzer.analyze_monthly_patterns(df, currency_code=currency)
        if 'monthly_stats' in monthly_result:
            print(f"   Monthly Pattern Analysis:")
            print(f"     - Strongest month: {monthly_result['strongest_month']['month']} (index: {monthly_result['strongest_month']['seasonal_index']:.2f})")
            print(f"     - Weakest month: {monthly_result['weakest_month']['month']} (index: {monthly_result['weakest_month']['seasonal_index']:.2f})")
            print(f"     - Seasonal variation: {monthly_result['seasonal_variation']:.4f}")
            print(f"     - Overall mean: {monthly_result['overall_mean']:.4f}")
        else:
            print(f"   {monthly_result['message']}")
        
        # Quarterly patterns
        quarterly_result = analyzer.analyze_quarterly_patterns(df, currency_code=currency)
        if 'quarterly_stats' in quarterly_result:
            print(f"   Quarterly Pattern Analysis:")
            print(f"     - Strongest quarter: {quarterly_result['strongest_quarter']['quarter']} (index: {quarterly_result['strongest_quarter']['seasonal_index']:.2f})")
            print(f"     - Weakest quarter: {quarterly_result['weakest_quarter']['quarter']} (index: {quarterly_result['weakest_quarter']['seasonal_index']:.2f})")
            print(f"     - Seasonal variation: {quarterly_result['seasonal_variation']:.4f}")
        else:
            print(f"   {quarterly_result['message']}")
        
        # Weekly patterns
        weekly_result = analyzer.analyze_weekly_patterns(df, currency_code=currency)
        if 'daily_stats' in weekly_result:
            print(f"   Weekly Pattern Analysis:")
            print(f"     - Strongest day: {weekly_result['strongest_day']['day_name']} (index: {weekly_result['strongest_day']['seasonal_index']:.2f})")
            print(f"     - Weakest day: {weekly_result['weakest_day']['day_name']} (index: {weekly_result['weakest_day']['seasonal_index']:.2f})")
            print(f"     - Seasonal variation: {weekly_result['seasonal_variation']:.4f}")
        else:
            print(f"   {weekly_result['message']}")
        
        # Periodicity detection
        periodicity_result = analyzer.detect_periodicity(df, currency_code=currency)
        if 'dominant_frequencies' in periodicity_result:
            print(f"   Periodicity Detection:")
            print(f"     - Dominant frequencies: {[f'{f:.4f}' for f in periodicity_result['dominant_frequencies']]}")
            print(f"     - Corresponding periods: {[f'{p:.2f}' for p in periodicity_result['periods']]}")
            print(f"     - Significant peaks: {len(periodicity_result['significant_peaks'])}")
        else:
            print(f"   {periodicity_result['message']}")
        
        # Stationarity test
        stationarity_result = analyzer.test_stationarity(df, currency_code=currency)
        if 'is_stationary' in stationarity_result:
            print(f"   Stationarity Test:")
            print(f"     - ADF Statistic: {stationarity_result['adf_statistic']:.4f}")
            print(f"     - P-value: {stationarity_result['p_value']:.4f}")
            print(f"     - Is stationary: {stationarity_result['is_stationary']}")
            print(f"     - Significance level: {stationarity_result['significance_level']}")
        else:
            print(f"   {stationarity_result['message']}")
    
    # Comprehensive analysis
    print("\n3. Running comprehensive seasonality analysis...")
    comprehensive_results = analyzer.analyze_all_seasonality(df)
    print(f"   ✓ Analyzed {comprehensive_results['summary']['total_currencies']} currencies")
    print(f"   ✓ Analysis completed at: {comprehensive_results['summary']['analysis_date']}")
    
    # Summary statistics
    print("\n4. Seasonality Summary Statistics:")
    for currency, results in comprehensive_results.items():
        if currency not in ['summary', 'overall_market']:
            seasonal_patterns = results['seasonal_patterns']
            if 'seasonal_strength' in seasonal_patterns:
                print(f"   {currency}:")
                print(f"     - Seasonal Strength: {seasonal_patterns['seasonal_strength']:.4f}")
                print(f"     - Seasonality Level: {'High' if seasonal_patterns['seasonal_strength'] > 0.6 else 'Medium' if seasonal_patterns['seasonal_strength'] > 0.3 else 'Low'}")
    
    print("\n=== Seasonality Analysis Example completed successfully! ===")
    print("Check the dashboard at http://localhost:8050 for interactive visualizations.")


if __name__ == "__main__":
    main()
