"""
Unit tests for data schema validation module
"""

import pytest
import json
import tempfile
import os
from src.data.schema import DataValidator, CurrencyData, Currency, ExchangeRate


class TestDataValidator:
    """Test cases for DataValidator class"""

    def test_valid_json_structure(self):
        """Test validation of valid JSON structure"""
        valid_data = {
            "success": True,
            "message": "Success",
            "data": []
        }
        assert DataValidator.validate_json_structure(valid_data) is True

    def test_invalid_json_structure_missing_field(self):
        """Test validation of JSON structure with missing field"""
        invalid_data = {
            "success": True,
            "data": []
        }
        assert DataValidator.validate_json_structure(invalid_data) is False

    def test_invalid_json_structure_wrong_data_type(self):
        """Test validation of JSON structure with wrong data type"""
        invalid_data = {
            "success": True,
            "message": "Success",
            "data": "not a list"
        }
        assert DataValidator.validate_json_structure(invalid_data) is False

    def test_valid_currency_structure(self):
        """Test validation of valid currency structure"""
        valid_currency = {
            "id": 1,
            "nameFr": "Dollar US",
            "nameAr": "الدولار الأمريكي",
            "unity": 1,
            "code": "USD",
            "exchangeRates": []
        }
        assert DataValidator.validate_currency_structure(valid_currency) is True

    def test_invalid_currency_structure_missing_field(self):
        """Test validation of currency structure with missing field"""
        invalid_currency = {
            "id": 1,
            "nameFr": "Dollar US",
            "nameAr": "الدولار الأمريكي",
            "unity": 1,
            "code": "USD"
        }
        assert DataValidator.validate_currency_structure(invalid_currency) is False

    def test_valid_exchange_rate_structure(self):
        """Test validation of valid exchange rate structure"""
        valid_rate = {
            "id": 137058,
            "day": "2016-06-14",
            "value": "333.21",
            "endDate": "2016-06-15"
        }
        assert DataValidator.validate_exchange_rate_structure(valid_rate) is True

    def test_invalid_exchange_rate_structure_missing_field(self):
        """Test validation of exchange rate structure with missing field"""
        invalid_rate = {
            "id": 137058,
            "day": "2016-06-14",
            "value": "333.21"
        }
        assert DataValidator.validate_exchange_rate_structure(invalid_rate) is False

    def test_validate_complete_data_success(self):
        """Test successful validation of complete data"""
        complete_data = {
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
                }
            ]
        }
        
        result = DataValidator.validate_complete_data(complete_data)
        assert isinstance(result, CurrencyData)
        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0].code == "USD"

    def test_load_and_validate_from_file_success(self):
        """Test successful loading and validation from file"""
        # Create temporary file with valid data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
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
                    }
                ]
            }, f)
            temp_file_path = f.name

        try:
            result = DataValidator.load_and_validate_from_file(temp_file_path)
            assert isinstance(result, CurrencyData)
            assert result.success is True
        finally:
            os.unlink(temp_file_path)

    def test_load_and_validate_from_file_not_found(self):
        """Test loading from non-existent file"""
        with pytest.raises(FileNotFoundError):
            DataValidator.load_and_validate_from_file("non_existent_file.json")

    def test_load_and_validate_from_file_invalid_json(self):
        """Test loading from file with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                DataValidator.load_and_validate_from_file(temp_file_path)
        finally:
            os.unlink(temp_file_path)


class TestExchangeRate:
    """Test cases for ExchangeRate dataclass"""

    def test_valid_exchange_rate(self):
        """Test creation of valid exchange rate"""
        rate = ExchangeRate(
            id=137058,
            day="2016-06-14",
            value="333.21",
            end_date="2016-06-15"
        )
        assert rate.id == 137058
        assert rate.day == "2016-06-14"
        assert rate.value == "333.21"
        assert rate.end_date == "2016-06-15"

    def test_invalid_exchange_rate_id(self):
        """Test creation of exchange rate with invalid ID"""
        with pytest.raises(ValueError, match="Invalid exchange rate ID"):
            ExchangeRate(
                id=0,
                day="2016-06-14",
                value="333.21",
                end_date="2016-06-15"
            )

    def test_invalid_exchange_rate_date(self):
        """Test creation of exchange rate with invalid date"""
        with pytest.raises(ValueError, match="Invalid day format"):
            ExchangeRate(
                id=137058,
                day="invalid-date",
                value="333.21",
                end_date="2016-06-15"
            )

    def test_invalid_exchange_rate_value(self):
        """Test creation of exchange rate with invalid value"""
        with pytest.raises(ValueError, match="Invalid value format"):
            ExchangeRate(
                id=137058,
                day="2016-06-14",
                value="-100",
                end_date="2016-06-15"
            )


class TestCurrency:
    """Test cases for Currency dataclass"""

    def test_valid_currency(self):
        """Test creation of valid currency"""
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
        
        assert currency.id == 1
        assert currency.name_fr == "Dollar US"
        assert currency.code == "USD"
        assert len(currency.exchange_rates) == 1

    def test_invalid_currency_code(self):
        """Test creation of currency with invalid code"""
        exchange_rates = [
            ExchangeRate(
                id=137058,
                day="2016-06-14",
                value="333.21",
                end_date="2016-06-15"
            )
        ]
        
        with pytest.raises(ValueError, match="Invalid currency code"):
            Currency(
                id=1,
                name_fr="Dollar US",
                name_ar="الدولار الأمريكي",
                unity=1,
                code="US",  # Too short
                exchange_rates=exchange_rates
            )


class TestCurrencyData:
    """Test cases for CurrencyData dataclass"""

    def test_valid_currency_data(self):
        """Test creation of valid currency data"""
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
        
        currency_data = CurrencyData(
            success=True,
            message="Success",
            data=[currency]
        )
        
        assert currency_data.success is True
        assert currency_data.message == "Success"
        assert len(currency_data.data) == 1
