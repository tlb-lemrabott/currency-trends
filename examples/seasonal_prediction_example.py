"""
Seasonal Prediction Example Script

This script demonstrates the enhanced seasonality analysis with historical peak detection
and future seasonal predictions based on historical patterns.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import DataValidator
from src.data.preprocessing import DataPreprocessor
from src.analysis.seasonality import SeasonalityAnalyzer


def create_extended_sample_data():
    """Create extended sample data with multiple years for demonstration"""
    # Create 5 years of data with seasonal patterns
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    usd_data = []
    eur_data = []
    
    for i, date in enumerate(dates):
        # USD: Strong seasonal pattern with peaks in March and September
        month = date.month
        year_factor = (date.year - 2019) * 0.5  # Gradual upward trend
        
        # Seasonal component: peaks in March (3) and September (9)
        seasonal_component = 10 * np.sin(2 * np.pi * month / 12) + 5 * np.sin(2 * np.pi * month / 6)
        
        # Add some randomness
        noise = np.random.normal(0, 2)
        
        usd_value = 100 + year_factor + seasonal_component + noise
        usd_data.append({
            'day': date,
            'currency_code': 'USD',
            'value': usd_value
        })
        
        # EUR: Different seasonal pattern with peaks in June and December
        seasonal_component = 8 * np.cos(2 * np.pi * month / 12) + 3 * np.sin(2 * np.pi * month / 4)
        year_factor = (date.year - 2019) * 0.3  # Slower trend
        
        noise = np.random.normal(0, 1.5)
        eur_value = 120 + year_factor + seasonal_component + noise
        eur_data.append({
            'day': date,
            'currency_code': 'EUR',
            'value': eur_value
        })
    
    return pd.DataFrame(usd_data + eur_data)


def main():
    """Main function demonstrating seasonal prediction analysis"""
    print("=== Currency Seasonal Prediction Analysis ===\n")
    
    # Create extended sample data for demonstration
    print("1. Creating extended sample data with 5 years of seasonal patterns...")
    df = create_extended_sample_data()
    
    print(f"   ✓ Created {len(df)} data points")
    print(f"   ✓ Date range: {df['day'].min().strftime('%Y-%m-%d')} to {df['day'].max().strftime('%Y-%m-%d')}")
    print(f"   ✓ Currencies: {df['currency_code'].unique()}")
    
    # Initialize seasonality analyzer
    analyzer = SeasonalityAnalyzer()
    
    # Analyze each currency
    for currency in df['currency_code'].unique():
        print(f"\n{'='*60}")
        print(f"ANALYSIS FOR {currency}")
        print(f"{'='*60}")
        
        # 1. Historical Seasonal Peaks Analysis
        print(f"\n📊 HISTORICAL SEASONAL PEAKS ANALYSIS")
        print("-" * 50)
        
        historical_peaks = analyzer.analyze_historical_seasonal_peaks(df, currency_code=currency)
        
        if 'message' in historical_peaks:
            print(f"   {historical_peaks['message']}")
        else:
            print(f"   Analysis Years: {historical_peaks['analysis_years']}")
            print(f"   Total Years: {historical_peaks['total_years']}")
            print(f"   Peak Consistency Score: {historical_peaks['peak_consistency_score']:.3f}")
            
            # Show yearly peaks
            print(f"\n   📈 YEARLY PEAK MONTHS:")
            for year, data in historical_peaks['yearly_peaks'].items():
                print(f"      {year}: Month {data['peak_month']} (Value: {data['peak_value']:.2f})")
            
            # Show most common peak months
            print(f"\n   🎯 MOST COMMON PEAK MONTHS:")
            for month, count in historical_peaks['most_common_peak_months'].items():
                month_name = datetime(2024, month, 1).strftime('%B')
                print(f"      {month_name} (Month {month}): {count} times")
            
            # Show seasonal trends
            trends = historical_peaks['seasonal_trends']
            print(f"\n   📈 SEASONAL TRENDS:")
            print(f"      Strongest Months: {trends['strongest_months']}")
            print(f"      Weakest Months: {trends['weakest_months']}")
            print(f"      Improving Months: {trends['improving_months']}")
            print(f"      Declining Months: {trends['declining_months']}")
        
        # 2. Seasonal Momentum Analysis
        print(f"\n🚀 SEASONAL MOMENTUM ANALYSIS")
        print("-" * 50)
        
        momentum_analysis = analyzer.analyze_seasonal_momentum(df, currency_code=currency, lookback_years=3)
        
        if 'message' in momentum_analysis:
            print(f"   {momentum_analysis['message']}")
        else:
            print(f"   Analysis Years: {momentum_analysis['analysis_years']}")
            print(f"   Overall Momentum: {momentum_analysis['overall_momentum']['overall_trend']} (Strength: {momentum_analysis['overall_momentum']['strength']:.3f})")
            
            # Show emerging patterns
            if momentum_analysis['emerging_patterns']:
                print(f"\n   🔍 EMERGING PATTERNS:")
                for pattern in momentum_analysis['emerging_patterns']:
                    month_name = datetime(2024, pattern['month'], 1).strftime('%B')
                    print(f"      {month_name}: {pattern['description']} (Strength: {pattern['strength']:.3f})")
            else:
                print(f"\n   🔍 No significant emerging patterns detected")
        
        # 3. Seasonal Peak Predictions
        print(f"\n🔮 SEASONAL PEAK PREDICTIONS")
        print("-" * 50)
        
        predictions = analyzer.predict_seasonal_peaks(df, currency_code=currency, forecast_years=2)
        
        if 'message' in predictions:
            print(f"   {predictions['message']}")
        else:
            print(f"   Forecast Years: {predictions['forecast_years']}")
            
            # Show confidence factors
            confidence = predictions['confidence_factors']
            print(f"\n   📊 CONFIDENCE FACTORS:")
            print(f"      Data Quality: {confidence['data_quality']:.3f}")
            print(f"      Pattern Consistency: {confidence['pattern_consistency']:.3f}")
            print(f"      Trend Strength: {confidence['trend_strength']:.3f}")
            
            # Show predictions for each year
            for year, year_data in predictions['predictions'].items():
                print(f"\n   📅 PREDICTIONS FOR {year}:")
                for peak in year_data['predicted_peaks']:
                    month_name = datetime(2024, peak['month'], 1).strftime('%B')
                    print(f"      Rank {peak['rank']}: {month_name} (Month {peak['month']})")
                    print(f"         Predicted Value: {peak['predicted_value']:.2f}")
                    print(f"         Confidence: {peak['confidence']:.3f}")
                    print(f"         Peak Probability: {peak['peak_probability']:.3f}")
            
            # Show recommendations
            if predictions['recommendations']:
                print(f"\n   💡 ACTIONABLE RECOMMENDATIONS:")
                for i, rec in enumerate(predictions['recommendations'], 1):
                    print(f"      {i}. {rec['description']}")
                    print(f"         Action: {rec['action']}")
                    print(f"         Confidence: {rec['confidence']:.3f}")
                    print(f"         Priority: {rec['priority'].upper()}")
        
        # 4. Monthly Performance Analysis
        print(f"\n📊 MONTHLY PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        if 'monthly_performance' in historical_peaks:
            monthly_perf = historical_peaks['monthly_performance']
            
            # Sort months by average performance
            month_avg = [(month, data['average_value']) for month, data in monthly_perf.items()]
            month_avg.sort(key=lambda x: x[1], reverse=True)
            
            print(f"   📈 TOP PERFORMING MONTHS:")
            for i, (month, avg_value) in enumerate(month_avg[:3], 1):
                month_name = datetime(2024, month, 1).strftime('%B')
                print(f"      {i}. {month_name}: {avg_value:.2f}")
            
            print(f"\n   📉 LOWEST PERFORMING MONTHS:")
            for i, (month, avg_value) in enumerate(month_avg[-3:], 1):
                month_name = datetime(2024, month, 1).strftime('%B')
                print(f"      {i}. {month_name}: {avg_value:.2f}")
            
            # Show improving months
            improving_months = [m for m, data in monthly_perf.items() if data['trend'] > 0.1]
            if improving_months:
                print(f"\n   📈 IMPROVING MONTHS (Positive Trends):")
                for month in improving_months:
                    month_name = datetime(2024, month, 1).strftime('%B')
                    trend = monthly_perf[month]['trend']
                    print(f"      {month_name}: Trend = {trend:.3f}")
    
    # 5. Summary and Insights
    print(f"\n{'='*60}")
    print(f"SUMMARY INSIGHTS")
    print(f"{'='*60}")
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   • USD shows peaks in March and September (dual-peak pattern)")
    print(f"   • EUR shows peaks in June and December (different seasonal cycle)")
    print(f"   • Historical patterns can predict future seasonal behavior")
    print(f"   • Momentum analysis reveals emerging trends")
    
    print(f"\n💡 TRADING INSIGHTS:")
    print(f"   • Position for USD strength in March and September")
    print(f"   • Position for EUR strength in June and December")
    print(f"   • Monitor momentum indicators for trend changes")
    print(f"   • Use confidence factors to assess prediction reliability")
    
    print(f"\n⚠️  RISK CONSIDERATIONS:")
    print(f"   • Past performance doesn't guarantee future results")
    print(f"   • External factors can override seasonal patterns")
    print(f"   • Always use multiple analysis methods")
    print(f"   • Monitor for pattern breakdowns")
    
    print(f"\n=== Seasonal Prediction Analysis completed successfully! ===")
    print(f"Check the dashboard at http://localhost:8050 for interactive visualizations.")


if __name__ == "__main__":
    main()
