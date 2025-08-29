"""
Unit tests for database module
"""

import pytest
import tempfile
import os
from src.data.database import DatabaseManager
from src.data.schema import CurrencyData, Currency, ExchangeRate, DataValidator


class TestDatabaseManager:
    """Test cases for DatabaseManager class"""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db_manager = DatabaseManager(db_path)
        yield db_manager
        
        # Cleanup
        os.unlink(db_path)

    def test_create_tables(self, temp_db):
        """Test that tables are created successfully"""
        # Tables should be created in __init__
        # Just verify we can connect and query
        currencies = temp_db.get_all_currencies()
        assert isinstance(currencies, list)

    def test_insert_and_get_currency(self, temp_db):
        """Test inserting and retrieving a currency"""
        # Create test currency
        exchange_rates = [
            ExchangeRate(
                id=137058,
                day="2016-06-14",
                value="333.21",
                end_date="2016-06-15"
            )
        ]
        
        currency = Currency(
            id=1,
            name_fr="Dollar US",
            name_ar="الدولار الأمريكي",
            unity=1,
            code="USD",
            exchange_rates=exchange_rates
        )
        
        # Insert currency
        result = temp_db.insert_currency(currency)
        assert result is True
        
        # Retrieve currency
        retrieved = temp_db.get_currency_by_code("USD")
        assert retrieved is not None
        assert retrieved['code'] == "USD"
        assert retrieved['name_fr'] == "Dollar US"

    def test_insert_and_get_exchange_rates(self, temp_db):
        """Test inserting and retrieving exchange rates"""
        # First insert currency
        exchange_rates = [
            ExchangeRate(
                id=137058,
                day="2016-06-14",
                value="333.21",
                end_date="2016-06-15"
            ),
            ExchangeRate(
                id=137059,
                day="2016-06-15",
                value="334.50",
                end_date="2016-06-16"
            )
        ]
        
        currency = Currency(
            id=1,
            name_fr="Dollar US",
            name_ar="الدولار الأمريكي",
            unity=1,
            code="USD",
            exchange_rates=exchange_rates
        )
        
        temp_db.insert_currency(currency)
        
        # Insert exchange rates
        for rate in exchange_rates:
            result = temp_db.insert_exchange_rate(currency.id, rate)
            assert result is True
        
        # Retrieve exchange rates
        rates = temp_db.get_exchange_rates_by_currency("USD")
        assert len(rates) == 2
        assert rates[0]['value'] == 333.21
        assert rates[1]['value'] == 334.50

    def test_insert_currency_data(self, temp_db):
        """Test inserting complete currency data"""
        # Create test data
        test_data = {
            "success": True,
            "message": "Success",
            "data": [
                {
                    "id": 1,
                    "nameFr": "Dollar US",
                    "nameAr": "الدولار الأمريكي",
                    "unity": 1,
                    "code": "USD",
                    "exchangeRates": [
                        {
                            "id": 137058,
                            "day": "2016-06-14",
                            "value": "333.21",
                            "endDate": "2016-06-15"
                        }
                    ]
                },
                {
                    "id": 2,
                    "nameFr": "Euro",
                    "nameAr": "يورو",
                    "unity": 1,
                    "code": "EUR",
                    "exchangeRates": [
                        {
                            "id": 137052,
                            "day": "2016-06-14",
                            "value": "423.21",
                            "endDate": "2016-06-15"
                        }
                    ]
                }
            ]
        }
        
        currency_data = DataValidator.validate_complete_data(test_data)
        
        # Insert complete data
        result = temp_db.insert_currency_data(currency_data)
        assert result is True
        
        # Verify currencies were inserted
        currencies = temp_db.get_all_currencies()
        assert len(currencies) == 2
        
        # Verify exchange rates were inserted
        usd_rates = temp_db.get_exchange_rates_by_currency("USD")
        eur_rates = temp_db.get_exchange_rates_by_currency("EUR")
        assert len(usd_rates) == 1
        assert len(eur_rates) == 1

    def test_get_exchange_rate_statistics(self, temp_db):
        """Test getting exchange rate statistics"""
        # Insert test data
        exchange_rates = [
            ExchangeRate(
                id=137058,
                day="2016-06-14",
                value="333.21",
                end_date="2016-06-15"
            ),
            ExchangeRate(
                id=137059,
                day="2016-06-15",
                value="334.50",
                end_date="2016-06-16"
            ),
            ExchangeRate(
                id=137060,
                day="2016-06-16",
                value="335.00",
                end_date="2016-06-17"
            )
        ]
        
        currency = Currency(
            id=1,
            name_fr="Dollar US",
            name_ar="الدولار الأمريكي",
            unity=1,
            code="USD",
            exchange_rates=exchange_rates
        )
        
        temp_db.insert_currency(currency)
        for rate in exchange_rates:
            temp_db.insert_exchange_rate(currency.id, rate)
        
        # Get statistics
        stats = temp_db.get_exchange_rate_statistics("USD")
        assert stats['count'] == 3
        assert stats['min_value'] == 333.21
        assert stats['max_value'] == 335.00
        assert stats['first_date'] == "2016-06-14"
        assert stats['last_date'] == "2016-06-16"

    def test_get_exchange_rates_with_date_filter(self, temp_db):
        """Test getting exchange rates with date filters"""
        # Insert test data
        exchange_rates = [
            ExchangeRate(
                id=137058,
                day="2016-06-14",
                value="333.21",
                end_date="2016-06-15"
            ),
            ExchangeRate(
                id=137059,
                day="2016-06-15",
                value="334.50",
                end_date="2016-06-16"
            ),
            ExchangeRate(
                id=137060,
                day="2016-06-16",
                value="335.00",
                end_date="2016-06-17"
            )
        ]
        
        currency = Currency(
            id=1,
            name_fr="Dollar US",
            name_ar="الدولار الأمريكي",
            unity=1,
            code="USD",
            exchange_rates=exchange_rates
        )
        
        temp_db.insert_currency(currency)
        for rate in exchange_rates:
            temp_db.insert_exchange_rate(currency.id, rate)
        
        # Test with start date filter
        rates = temp_db.get_exchange_rates_by_currency("USD", start_date="2016-06-15")
        assert len(rates) == 2
        
        # Test with end date filter
        rates = temp_db.get_exchange_rates_by_currency("USD", end_date="2016-06-15")
        assert len(rates) == 2
        
        # Test with both filters
        rates = temp_db.get_exchange_rates_by_currency("USD", 
                                                      start_date="2016-06-15", 
                                                      end_date="2016-06-15")
        assert len(rates) == 1

    def test_delete_currency_data(self, temp_db):
        """Test deleting currency data"""
        # Insert test data
        exchange_rates = [
            ExchangeRate(
                id=137058,
                day="2016-06-14",
                value="333.21",
                end_date="2016-06-15"
            )
        ]
        
        currency = Currency(
            id=1,
            name_fr="Dollar US",
            name_ar="الدولار الأمريكي",
            unity=1,
            code="USD",
            exchange_rates=exchange_rates
        )
        
        temp_db.insert_currency(currency)
        temp_db.insert_exchange_rate(currency.id, exchange_rates[0])
        
        # Verify data exists
        assert temp_db.get_currency_by_code("USD") is not None
        assert len(temp_db.get_exchange_rates_by_currency("USD")) == 1
        
        # Delete currency data
        result = temp_db.delete_currency_data("USD")
        assert result is True
        
        # Verify data is deleted
        assert temp_db.get_currency_by_code("USD") is None
        assert len(temp_db.get_exchange_rates_by_currency("USD")) == 0

    def test_get_database_info(self, temp_db):
        """Test getting database information"""
        # Insert some test data
        exchange_rates = [
            ExchangeRate(
                id=137058,
                day="2016-06-14",
                value="333.21",
                end_date="2016-06-15"
            )
        ]
        
        currency = Currency(
            id=1,
            name_fr="Dollar US",
            name_ar="الدولار الأمريكي",
            unity=1,
            code="USD",
            exchange_rates=exchange_rates
        )
        
        temp_db.insert_currency(currency)
        temp_db.insert_exchange_rate(currency.id, exchange_rates[0])
        
        # Get database info
        info = temp_db.get_database_info()
        assert info['currency_count'] == 1
        assert info['exchange_rate_count'] == 1
        assert info['earliest_date'] == "2016-06-14"
        assert info['latest_date'] == "2016-06-14"
        assert info['database_path'] == temp_db.db_path

    def test_get_nonexistent_currency(self, temp_db):
        """Test getting a currency that doesn't exist"""
        result = temp_db.get_currency_by_code("NONEXISTENT")
        assert result is None

    def test_get_exchange_rates_nonexistent_currency(self, temp_db):
        """Test getting exchange rates for a currency that doesn't exist"""
        rates = temp_db.get_exchange_rates_by_currency("NONEXISTENT")
        assert rates == []

    def test_delete_nonexistent_currency(self, temp_db):
        """Test deleting a currency that doesn't exist"""
        result = temp_db.delete_currency_data("NONEXISTENT")
        assert result is False
