"""
Data Ingestion Example Script

This script demonstrates how to use the data ingestion pipeline to load
currency exchange rate data from various sources.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import DatabaseManager
from src.data.ingestion import DataIngestionPipeline


def main():
    """Main function demonstrating ingestion pipeline operations"""
    print("=== Currency Trends Data Ingestion Example ===\n")
    
    # Initialize database and pipeline
    db_manager = DatabaseManager("ingestion_example.db")
    pipeline = DataIngestionPipeline(db_manager)
    
    # Example 1: Ingest from file
    print("1. Ingesting data from file...")
    result = pipeline.ingest_from_file('data/sample_currency_data.json')
    
    if result['success']:
        print(f"   ✓ Successfully processed {result['currencies_processed']} currencies")
        print(f"   ✓ Processed {result['rates_processed']} exchange rates")
    else:
        print(f"   ✗ Failed: {result['errors']}")
    
    # Example 2: Validate data quality
    print("\n2. Validating data quality...")
    from src.data.schema import DataValidator
    currency_data = DataValidator.load_and_validate_from_file('data/sample_currency_data.json')
    quality_report = pipeline.validate_data_quality(currency_data)
    
    print(f"   - Total currencies: {quality_report['total_currencies']}")
    print(f"   - Total rates: {quality_report['total_rates']}")
    print(f"   - Date range: {quality_report['date_range']['earliest']} to {quality_report['date_range']['latest']}")
    print(f"   - Quality score: {quality_report['quality_score']:.2f}")
    
    # Example 3: Get ingestion status
    print("\n3. Current database status:")
    status = pipeline.get_ingestion_status()
    print(f"   - Currencies in database: {status['currency_count']}")
    print(f"   - Exchange rates in database: {status['exchange_rate_count']}")
    print(f"   - Date range: {status['earliest_date']} to {status['latest_date']}")
    
    # Example 4: Demonstrate incremental update
    print("\n4. Demonstrating incremental update...")
    
    # Create new data with additional rates
    new_data = {
        "success": True,
        "message": "Incremental update",
        "data": [
            {
                "id": 1,
                "nameFr": "Dollar US",
                "nameAr": "الدولار الأمريكي",
                "unity": 1,
                "code": "USD",
                "exchangeRates": [
                    {
                        "id": 173095,
                        "day": "2025-08-22",
                        "value": "395.50",
                        "endDate": "2025-08-23"
                    }
                ]
            }
        ]
    }
    
    update_result = pipeline.ingest_incremental_update(new_data)
    
    if update_result['success']:
        print(f"   ✓ Updated {update_result['currencies_updated']} currencies")
        print(f"   ✓ Added {update_result['rates_added']} new rates")
    else:
        print(f"   ✗ Update failed: {update_result['errors']}")
    
    # Example 5: Show updated status
    print("\n5. Updated database status:")
    updated_status = pipeline.get_ingestion_status()
    print(f"   - Currencies in database: {updated_status['currency_count']}")
    print(f"   - Exchange rates in database: {updated_status['exchange_rate_count']}")
    print(f"   - Date range: {updated_status['earliest_date']} to {updated_status['latest_date']}")
    
    # Example 6: Retrieve specific currency data
    print("\n6. Retrieving USD exchange rates:")
    usd_rates = db_manager.get_exchange_rates_by_currency("USD")
    for rate in usd_rates:
        print(f"   - {rate['day']}: {rate['value']}")
    
    print("\n=== Ingestion Example completed successfully! ===")
    print("Database file: ingestion_example.db")


if __name__ == "__main__":
    main()
