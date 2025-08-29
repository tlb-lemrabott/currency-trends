#!/usr/bin/env python3
"""
Test script to verify dashboard seasonal predictions functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_seasonal_predictions():
    """Test the seasonal predictions functionality"""
    print("Testing Seasonal Predictions Functionality...")
    
    try:
        # Test 1: Import the seasonality analyzer
        from src.analysis.seasonality import SeasonalityAnalyzer
        print("✓ SeasonalityAnalyzer imported successfully")
        
        # Test 2: Create sample data
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2024, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        sample_data = []
        for date in dates:
            month = date.month
            year_factor = (date.year - 2019) * 0.5
            seasonal_component = 10 * np.sin(2 * np.pi * month / 12) + 5 * np.sin(2 * np.pi * month / 6)
            noise = np.random.normal(0, 2)
            usd_value = 100 + year_factor + seasonal_component + noise
            
            sample_data.append({
                'day': date,
                'currency_code': 'USD',
                'value': usd_value
            })
        
        df = pd.DataFrame(sample_data)
        print(f"✓ Sample data created: {len(df)} data points")
        
        # Test 3: Test seasonal predictions
        analyzer = SeasonalityAnalyzer()
        predictions = analyzer.predict_seasonal_peaks(df, currency_code='USD', forecast_years=2)
        print("✓ Seasonal predictions generated successfully")
        
        # Test 4: Test historical peaks
        historical_peaks = analyzer.analyze_historical_seasonal_peaks(df, currency_code='USD')
        print("✓ Historical peaks analysis completed")
        
        # Test 5: Test seasonal momentum
        momentum = analyzer.analyze_seasonal_momentum(df, currency_code='USD', lookback_years=3)
        print("✓ Seasonal momentum analysis completed")
        
        # Test 6: Test dashboard chart creation
        from src.visualization.dashboard import CurrencyDashboard
        dashboard = CurrencyDashboard()
        
        # Test the seasonal predictions chart method
        chart = dashboard.create_seasonal_predictions_chart(df, 'USD')
        print("✓ Dashboard seasonal predictions chart created successfully")
        
        print("\n🎉 All tests passed! Seasonal predictions functionality is working correctly.")
        
        # Show some results
        print(f"\n📊 Sample Results:")
        print(f"   - Forecast Years: {predictions.get('forecast_years', 'N/A')}")
        print(f"   - Total Years Analyzed: {historical_peaks.get('total_years', 'N/A')}")
        print(f"   - Peak Consistency Score: {historical_peaks.get('peak_consistency_score', 'N/A'):.3f}")
        
        if 'predictions' in predictions:
            for year, year_data in predictions['predictions'].items():
                print(f"   - {year} Top Predicted Peak: Month {year_data['predicted_peaks'][0]['month']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing seasonal predictions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_seasonal_predictions()
    if success:
        print("\n✅ Ready to test in dashboard!")
        print("1. Open http://localhost:8050 in your browser")
        print("2. Select 'Seasonal Predictions' from the Analysis Type dropdown")
        print("3. Click 'Update Analysis' button")
        print("4. You should see the seasonal predictions chart!")
    else:
        print("\n❌ There are issues that need to be fixed before testing in dashboard.")
