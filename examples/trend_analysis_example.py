"""
Trend Analysis Example Script

This script demonstrates the trend analysis functionality with concrete results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import DataValidator
from src.data.preprocessing import DataPreprocessor
from src.analysis.trends import TrendAnalyzer


def main():
    """Main function demonstrating trend analysis"""
    print("=== Currency Trends Analysis Example ===\n")
    
    # Load and preprocess data
    print("1. Loading and preprocessing data...")
    currency_data = DataValidator.load_and_validate_from_file('data/sample_currency_data.json')
    preprocessor = DataPreprocessor()
    df = preprocessor.convert_to_dataframe(currency_data)
    
    # Add technical indicators
    df = preprocessor.add_technical_indicators(df)
    df = preprocessor.add_time_features(df)
    
    print(f"   ✓ Loaded {len(currency_data.data)} currencies")
    print(f"   ✓ DataFrame shape: {df.shape}")
    
    # Initialize trend analyzer
    analyzer = TrendAnalyzer()
    
    # Analyze trends for each currency
    print("\n2. Analyzing trends for each currency...")
    
    for currency in df['currency_code'].unique():
        print(f"\n--- {currency} Analysis ---")
        
        # Linear trend analysis
        linear_trend = analyzer.calculate_linear_trend(df, currency)
        print(f"   Linear Trend:")
        print(f"     - Direction: {linear_trend['trend_direction']}")
        print(f"     - Slope: {linear_trend['slope']:.4f}")
        print(f"     - R² Score: {linear_trend['r2_score']:.4f}")
        print(f"     - Total Change: {linear_trend['total_change_percent']:.2f}%")
        print(f"     - Volatility: {linear_trend['volatility']:.2f}")
        
        # Trend strength analysis
        trend_strength = analyzer.calculate_trend_strength(df, currency)
        print(f"   Trend Strength:")
        print(f"     - Category: {trend_strength['strength_category']}")
        print(f"     - Direction: {trend_strength['trend_direction']}")
        print(f"     - Consistency: {trend_strength['trend_consistency']:.2f}")
        print(f"     - Momentum: {trend_strength['momentum']:.4f}")
        print(f"     - Positive Days: {trend_strength['positive_days']}")
        print(f"     - Negative Days: {trend_strength['negative_days']}")
        
        # Moving averages
        ma_results = analyzer.calculate_moving_averages(df, currency_code=currency)
        print(f"   Moving Averages:")
        print(f"     - Data points: {len(ma_results['actual_values'])}")
        print(f"     - Available MAs: {list(ma_results['moving_averages'].keys())}")
        
        # Trend changes
        trend_changes = analyzer.detect_trend_changes(df, currency_code=currency)
        print(f"   Trend Changes:")
        print(f"     - Total changes: {trend_changes['total_changes']}")
        print(f"     - Bullish changes: {trend_changes['bullish_changes']}")
        print(f"     - Bearish changes: {trend_changes['bearish_changes']}")
        
        if trend_changes['trend_changes']:
            print("     - Recent changes:")
            for change in trend_changes['trend_changes'][-3:]:  # Show last 3 changes
                print(f"       * {change['date']}: {change['type']} (Value: {change['value']:.2f})")
    
    # Overall market analysis
    print("\n3. Overall Market Analysis...")
    overall_trend = analyzer.calculate_linear_trend(df)
    print(f"   Overall Market Trend:")
    print(f"     - Direction: {overall_trend['trend_direction']}")
    print(f"     - Total Change: {overall_trend['total_change_percent']:.2f}%")
    print(f"     - R² Score: {overall_trend['r2_score']:.4f}")
    
    # Comprehensive analysis
    print("\n4. Running comprehensive analysis...")
    comprehensive_results = analyzer.analyze_all_trends(df)
    print(f"   ✓ Analyzed {comprehensive_results['summary']['total_currencies']} currencies")
    print(f"   ✓ Analysis completed at: {comprehensive_results['summary']['analysis_date']}")
    
    # Summary statistics
    print("\n5. Summary Statistics:")
    for currency, results in comprehensive_results['currencies'].items():
        linear = results['linear_trend']
        strength = results['trend_strength']
        print(f"   {currency}:")
        print(f"     - Trend: {linear['trend_direction']} ({linear['total_change_percent']:.2f}%)")
        print(f"     - Strength: {strength['strength_category']}")
        print(f"     - Data Points: {linear['data_points']}")
    
    print("\n=== Trend Analysis Example completed successfully! ===")
    print("Check the dashboard at http://localhost:8050 for interactive visualizations.")


if __name__ == "__main__":
    main()
