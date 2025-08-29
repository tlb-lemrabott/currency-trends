"""
Database Example Script

This script demonstrates how to use the database module to store and retrieve
currency exchange rate data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import DatabaseManager
from src.data.schema import DataValidator


def main():
    """Main function demonstrating database operations"""
    print("=== Currency Trends Database Example ===\n")
    
    # Initialize database manager
    db_manager = DatabaseManager("example_currency_trends.db")
    
    # Load sample data
    print("1. Loading sample currency data...")
    try:
        currency_data = DataValidator.load_and_validate_from_file('data/sample_currency_data.json')
        print(f"   ✓ Loaded {len(currency_data.data)} currencies")
        for currency in currency_data.data:
            print(f"   - {currency.name_fr} ({currency.code}): {len(currency.exchange_rates)} rates")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # Insert data into database
    print("\n2. Inserting data into database...")
    success = db_manager.insert_currency_data(currency_data)
    if success:
        print("   ✓ Data inserted successfully")
    else:
        print("   ✗ Error inserting data")
        return
    
    # Get database information
    print("\n3. Database information:")
    info = db_manager.get_database_info()
    print(f"   - Currencies: {info['currency_count']}")
    print(f"   - Exchange rates: {info['exchange_rate_count']}")
    print(f"   - Date range: {info['earliest_date']} to {info['latest_date']}")
    
    # Retrieve specific currency data
    print("\n4. Retrieving USD exchange rates:")
    usd_rates = db_manager.get_exchange_rates_by_currency("USD")
    for rate in usd_rates:
        print(f"   - {rate['day']}: {rate['value']}")
    
    # Get statistics
    print("\n5. USD Statistics:")
    stats = db_manager.get_exchange_rate_statistics("USD")
    print(f"   - Count: {stats['count']}")
    print(f"   - Min: {stats['min_value']}")
    print(f"   - Max: {stats['max_value']}")
    print(f"   - Average: {stats['avg_value']:.2f}")
    
    # Test date filtering
    print("\n6. USD rates from 2016-06-15 onwards:")
    filtered_rates = db_manager.get_exchange_rates_by_currency("USD", start_date="2016-06-15")
    for rate in filtered_rates:
        print(f"   - {rate['day']}: {rate['value']}")
    
    # Get all currencies
    print("\n7. All currencies in database:")
    currencies = db_manager.get_all_currencies()
    for currency in currencies:
        print(f"   - {currency['name_fr']} ({currency['code']})")
    
    print("\n=== Example completed successfully! ===")
    print("Database file: example_currency_trends.db")


if __name__ == "__main__":
    main()
