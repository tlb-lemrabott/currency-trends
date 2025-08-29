"""
Data Preprocessing Example Script

This script demonstrates how to use the data preprocessing module to clean
and prepare currency exchange rate data for analysis.
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import DataPreprocessor
from src.data.schema import DataValidator


def main():
    """Main function demonstrating preprocessing operations"""
    print("=== Currency Trends Data Preprocessing Example ===\n")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load sample data
    print("1. Loading sample currency data...")
    currency_data = DataValidator.load_and_validate_from_file('data/sample_currency_data.json')
    print(f"   ✓ Loaded {len(currency_data.data)} currencies")
    
    # Convert to DataFrame
    print("\n2. Converting to DataFrame...")
    df = preprocessor.convert_to_dataframe(currency_data)
    print(f"   ✓ DataFrame shape: {df.shape}")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    # Handle missing values (if any)
    print("\n3. Handling missing values...")
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"   - Found {missing_count} missing values")
        df_clean = preprocessor.handle_missing_values(df, method='interpolate')
        print(f"   ✓ Missing values handled using interpolation")
    else:
        print("   - No missing values found")
        df_clean = df.copy()
    
    # Detect and handle outliers
    print("\n4. Detecting and handling outliers...")
    df_clean = preprocessor.detect_and_handle_outliers(df_clean, method='iqr', threshold=1.5)
    print("   ✓ Outliers handled using IQR method")
    
    # Normalize values
    print("\n5. Normalizing values...")
    df_normalized = preprocessor.normalize_values(df_clean, method='minmax')
    print("   ✓ Values normalized using min-max method")
    
    # Add technical indicators
    print("\n6. Adding technical indicators...")
    df_with_indicators = preprocessor.add_technical_indicators(df_normalized)
    print("   ✓ Technical indicators added:")
    indicators = ['sma_7', 'sma_30', 'ema_12', 'volatility_7', 'volatility_30', 
                 'price_change', 'price_change_pct', 'bb_upper', 'bb_lower', 'bb_position']
    for indicator in indicators:
        if indicator in df_with_indicators.columns:
            print(f"     - {indicator}")
    
    # Add time features
    print("\n7. Adding time features...")
    df_with_time = preprocessor.add_time_features(df_with_indicators)
    print("   ✓ Time features added:")
    time_features = ['year', 'month', 'day_of_week', 'quarter', 'month_sin', 'month_cos']
    for feature in time_features:
        if feature in df_with_time.columns:
            print(f"     - {feature}")
    
    # Create lagged features
    print("\n8. Creating lagged features...")
    df_with_lags = preprocessor.create_lagged_features(df_with_time, lags=[1, 7])
    print("   ✓ Lagged features created:")
    lag_features = ['value_lag_1', 'value_lag_7', 'price_change_lag_1', 'price_change_lag_7']
    for feature in lag_features:
        if feature in df_with_lags.columns:
            print(f"     - {feature}")
    
    # Generate preprocessing summary
    print("\n9. Preprocessing summary:")
    summary = preprocessor.get_preprocessing_summary(df, df_with_lags)
    print(f"   - Original shape: {summary['original_shape']}")
    print(f"   - Processed shape: {summary['processed_shape']}")
    print(f"   - Currencies processed: {summary['currencies_processed']}")
    print(f"   - Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"   - Features added: {len(summary['features_added'])}")
    
    # Demonstrate complete preprocessing pipeline
    print("\n10. Complete preprocessing pipeline...")
    config = {
        'handle_missing': 'interpolate',
        'handle_outliers': 'iqr',
        'outlier_threshold': 1.5,
        'normalize': 'minmax',
        'add_indicators': True,
        'add_time_features': True,
        'lags': [1, 7, 30]
    }
    
    df_complete, summary_complete = preprocessor.preprocess_complete(currency_data, config)
    print(f"   ✓ Complete pipeline executed")
    print(f"   - Final shape: {df_complete.shape}")
    print(f"   - Total features: {len(df_complete.columns)}")
    
    # Show sample of processed data
    print("\n11. Sample of processed data:")
    print(df_complete.head(3).to_string())
    
    # Show feature statistics
    print("\n12. Feature statistics:")
    numeric_columns = df_complete.select_dtypes(include=[np.number]).columns
    for col in numeric_columns[:5]:  # Show first 5 numeric columns
        if col in df_complete.columns:
            print(f"   - {col}: mean={df_complete[col].mean():.4f}, std={df_complete[col].std():.4f}")
    
    print("\n=== Preprocessing Example completed successfully! ===")
    print("The data is now ready for analysis and modeling.")


if __name__ == "__main__":
    main()
