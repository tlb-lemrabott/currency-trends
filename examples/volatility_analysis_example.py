"""
Volatility Analysis Example Script

This script demonstrates the volatility analysis functionality with concrete results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import DataValidator
from src.data.preprocessing import DataPreprocessor
from src.analysis.volatility import VolatilityAnalyzer


def main():
    """Main function demonstrating volatility analysis"""
    print("=== Currency Volatility Analysis Example ===\n")
    
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
    
    # Initialize volatility analyzer
    analyzer = VolatilityAnalyzer()
    
    # Analyze volatility for each currency
    print("\n2. Analyzing volatility for each currency...")
    
    for currency in df['currency_code'].unique():
        print(f"\n--- {currency} Volatility Analysis ---")
        
        # Rolling volatility analysis
        rolling_vol = analyzer.calculate_rolling_volatility(df, currency_code=currency)
        print(f"   Rolling Volatility (20-day window):")
        print(f"     - Data points: {len(rolling_vol['volatility'])}")
        print(f"     - Annualized: {rolling_vol['annualized']}")
        if rolling_vol['volatility']:
            valid_vol = [v for v in rolling_vol['volatility'] if v is not None]
            if valid_vol:
                print(f"     - Current volatility: {valid_vol[-1]:.4f}")
                print(f"     - Average volatility: {sum(valid_vol)/len(valid_vol):.4f}")
        
        # Volatility metrics
        vol_metrics = analyzer.calculate_volatility_metrics(df, currency_code=currency)
        if 'annualized_volatility' in vol_metrics:
            print(f"   Volatility Metrics:")
            print(f"     - Annualized volatility: {vol_metrics['annualized_volatility']:.4f}")
            print(f"     - Daily volatility: {vol_metrics['daily_volatility']:.4f}")
            print(f"     - Min volatility: {vol_metrics['min_volatility']:.4f}")
            print(f"     - Max volatility: {vol_metrics['max_volatility']:.4f}")
            print(f"     - Skewness: {vol_metrics['skewness']:.4f}")
            print(f"     - Kurtosis: {vol_metrics['kurtosis']:.4f}")
            print(f"     - VaR (95%): {vol_metrics['var_95']:.4f}")
            print(f"     - CVaR (95%): {vol_metrics['cvar_95']:.4f}")
        
        # Volatility regime analysis
        vol_regime = analyzer.calculate_volatility_regime(df, currency_code=currency)
        print(f"   Volatility Regime Analysis:")
        print(f"     - High volatility periods: {vol_regime['high_vol_periods']}")
        print(f"     - Low volatility periods: {vol_regime['low_vol_periods']}")
        print(f"     - Current regime: {vol_regime['current_regime']}")
        if vol_regime['current_volatility']:
            print(f"     - Current volatility: {vol_regime['current_volatility']:.4f}")
        
        # Historical volatility
        hist_vol = analyzer.calculate_historical_volatility(df, currency_code=currency)
        print(f"   Historical Volatility:")
        print(f"     - Available periods: {list(hist_vol['historical_volatility'].keys())}")
        
        # Volatility forecast
        vol_forecast = analyzer.calculate_volatility_forecast(df, currency_code=currency)
        print(f"   Volatility Forecast:")
        print(f"     - Forecast days: {vol_forecast['forecast_days']}")
        if vol_forecast['forecast']:
            print(f"     - Current volatility: {vol_forecast['current_volatility']:.4f}")
            print(f"     - Average forecast: {sum(vol_forecast['forecast'])/len(vol_forecast['forecast']):.4f}")
    
    # Overall market volatility analysis
    print("\n3. Overall Market Volatility Analysis...")
    overall_metrics = analyzer.calculate_volatility_metrics(df)
    if 'annualized_volatility' in overall_metrics:
        print(f"   Overall Market Volatility:")
        print(f"     - Annualized volatility: {overall_metrics['annualized_volatility']:.4f}")
        print(f"     - Daily volatility: {overall_metrics['daily_volatility']:.4f}")
        print(f"     - Data points: {overall_metrics['data_points']}")
    
    # Comprehensive analysis
    print("\n4. Running comprehensive volatility analysis...")
    comprehensive_results = analyzer.analyze_all_volatility(df)
    print(f"   ✓ Analyzed {comprehensive_results['summary']['total_currencies']} currencies")
    print(f"   ✓ Analysis completed at: {comprehensive_results['summary']['analysis_date']}")
    
    # Summary statistics
    print("\n5. Volatility Summary Statistics:")
    for currency, results in comprehensive_results['currencies'].items():
        metrics = results['volatility_metrics']
        if 'annualized_volatility' in metrics:
            print(f"   {currency}:")
            print(f"     - Annualized Volatility: {metrics['annualized_volatility']:.4f}")
            print(f"     - Risk Level: {'High' if metrics['annualized_volatility'] > 0.3 else 'Medium' if metrics['annualized_volatility'] > 0.15 else 'Low'}")
            print(f"     - Data Points: {metrics['data_points']}")
    
    print("\n=== Volatility Analysis Example completed successfully! ===")
    print("Check the dashboard at http://localhost:8050 for interactive visualizations.")


if __name__ == "__main__":
    main()
