"""
Data Schema Validation Module

This module handles validation of the currency exchange rate JSON data structure.
Implements single responsibility principle by focusing only on data validation.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExchangeRate:
    """Data class for exchange rate entries"""
    id: int
    day: str
    value: str
    end_date: str

    def __post_init__(self):
        """Validate exchange rate data after initialization"""
        if not isinstance(self.id, int) or self.id <= 0:
            raise ValueError(f"Invalid exchange rate ID: {self.id}")
        
        if not self._is_valid_date(self.day):
            raise ValueError(f"Invalid day format: {self.day}")
        
        if not self._is_valid_date(self.end_date):
            raise ValueError(f"Invalid end_date format: {self.end_date}")
        
        if not self._is_valid_value(self.value):
            raise ValueError(f"Invalid value format: {self.value}")

    def _is_valid_date(self, date_str: str) -> bool:
        """Validate date string format (YYYY-MM-DD)"""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def _is_valid_value(self, value: str) -> bool:
        """Validate exchange rate value format"""
        try:
            float_val = float(value)
            return float_val >= 0  # Allow zero values
        except (ValueError, TypeError):
            return False


@dataclass
class Currency:
    """Data class for currency entries"""
    id: int
    name_fr: str
    name_ar: str
    unity: int
    code: str
    exchange_rates: List[ExchangeRate]

    def __post_init__(self):
        """Validate currency data after initialization"""
        if not isinstance(self.id, int) or self.id <= 0:
            raise ValueError(f"Invalid currency ID: {self.id}")
        
        if not self.name_fr or not isinstance(self.name_fr, str):
            raise ValueError(f"Invalid French name: {self.name_fr}")
        
        if not self.name_ar or not isinstance(self.name_ar, str):
            raise ValueError(f"Invalid Arabic name: {self.name_ar}")
        
        if not isinstance(self.unity, int) or self.unity <= 0:
            raise ValueError(f"Invalid unity value: {self.unity}")
        
        if not self.code or not isinstance(self.code, str) or len(self.code) != 3:
            raise ValueError(f"Invalid currency code: {self.code}")
        
        if not isinstance(self.exchange_rates, list):
            raise ValueError("Exchange rates must be a list")


@dataclass
class CurrencyData:
    """Data class for the complete currency dataset"""
    success: bool
    message: str
    data: List[Currency]

    def __post_init__(self):
        """Validate complete dataset"""
        if not isinstance(self.success, bool):
            raise ValueError(f"Invalid success field: {self.success}")
        
        if not self.message or not isinstance(self.message, str):
            raise ValueError(f"Invalid message: {self.message}")
        
        if not isinstance(self.data, list):
            raise ValueError("Data must be a list")


class DataValidator:
    """Validator class for currency exchange rate data"""
    
    @staticmethod
    def detect_data_structure(json_data: Dict[str, Any]) -> str:
        """
        Detect the structure type of the JSON data
        
        Args:
            json_data: Dictionary containing the JSON data
            
        Returns:
            str: 'strapi' for Strapi-like structure, 'flat' for flat structure
        """
        if 'data' in json_data and isinstance(json_data['data'], list) and len(json_data['data']) > 0:
            first_item = json_data['data'][0]
            if 'attributes' in first_item:
                return 'strapi'
        return 'flat'
    
    @staticmethod
    def transform_strapi_to_flat(json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Strapi-like structure to flat structure
        
        Args:
            json_data: Dictionary containing Strapi-like JSON data
            
        Returns:
            Dict[str, Any]: Transformed flat structure
        """
        transformed_data = {
            'success': json_data.get('success', True),
            'message': json_data.get('message', ''),
            'data': []
        }
        
        for currency_item in json_data.get('data', []):
            currency_id = currency_item.get('id')
            attributes = currency_item.get('attributes', {})
            
            # Transform currency data
            currency_data = {
                'id': currency_id,
                'nameFr': attributes.get('name_fr', ''),
                'nameAr': attributes.get('name_ar', ''),
                'unity': attributes.get('unity', 1),
                'code': attributes.get('code', ''),
                'exchangeRates': []
            }
            
            # Transform exchange rates
            money_changes = attributes.get('money_today_changes', {})
            if 'data' in money_changes:
                for rate_item in money_changes['data']:
                    rate_id = rate_item.get('id')
                    rate_attributes = rate_item.get('attributes', {})
                    
                    rate_data = {
                        'id': rate_id,
                        'day': rate_attributes.get('day', ''),
                        'value': rate_attributes.get('value', ''),
                        'endDate': rate_attributes.get('end_date', '')
                    }
                    currency_data['exchangeRates'].append(rate_data)
            
            transformed_data['data'].append(currency_data)
        
        return transformed_data

    @staticmethod
    def validate_json_structure(json_data: Dict[str, Any]) -> bool:
        """
        Validate the overall JSON structure
        
        Args:
            json_data: Dictionary containing the JSON data
            
        Returns:
            bool: True if structure is valid, False otherwise
        """
        required_fields = ['success', 'message', 'data']
        
        # Check if all required fields exist
        for field in required_fields:
            if field not in json_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check if data is a list
        if not isinstance(json_data['data'], list):
            logger.error("Data field must be a list")
            return False
        
        return True

    @staticmethod
    def validate_currency_structure(currency_data: Dict[str, Any]) -> bool:
        """
        Validate individual currency structure
        
        Args:
            currency_data: Dictionary containing currency data
            
        Returns:
            bool: True if structure is valid, False otherwise
        """
        required_fields = ['id', 'nameFr', 'nameAr', 'unity', 'code', 'exchangeRates']
        
        # Check if all required fields exist
        for field in required_fields:
            if field not in currency_data:
                logger.error(f"Missing required currency field: {field}")
                return False
        
        # Check if exchangeRates is a list
        if not isinstance(currency_data['exchangeRates'], list):
            logger.error("Exchange rates must be a list")
            return False
        
        return True

    @staticmethod
    def validate_exchange_rate_structure(rate_data: Dict[str, Any]) -> bool:
        """
        Validate individual exchange rate structure
        
        Args:
            rate_data: Dictionary containing exchange rate data
            
        Returns:
            bool: True if structure is valid, False otherwise
        """
        required_fields = ['id', 'day', 'value', 'endDate']
        
        # Check if all required fields exist
        for field in required_fields:
            if field not in rate_data:
                logger.error(f"Missing required exchange rate field: {field}")
                return False
        
        return True

    @classmethod
    def validate_complete_data(cls, json_data: Dict[str, Any]) -> CurrencyData:
        """
        Validate and parse complete JSON data into structured objects
        
        Args:
            json_data: Dictionary containing the complete JSON data
            
        Returns:
            CurrencyData: Validated and parsed data structure
            
        Raises:
            ValueError: If data validation fails
        """
        # Detect and transform data structure if needed
        structure_type = cls.detect_data_structure(json_data)
        if structure_type == 'strapi':
            logger.info("Detected Strapi-like structure, transforming to flat structure")
            json_data = cls.transform_strapi_to_flat(json_data)
        
        # Validate overall structure
        if not cls.validate_json_structure(json_data):
            raise ValueError("Invalid JSON structure")
        
        # Parse exchange rates
        currencies = []
        for currency_dict in json_data['data']:
            if not cls.validate_currency_structure(currency_dict):
                raise ValueError(f"Invalid currency structure: {currency_dict.get('id', 'unknown')}")
            
            # Parse exchange rates for this currency
            exchange_rates = []
            for rate_dict in currency_dict['exchangeRates']:
                if not cls.validate_exchange_rate_structure(rate_dict):
                    raise ValueError(f"Invalid exchange rate structure: {rate_dict.get('id', 'unknown')}")
                
                exchange_rate = ExchangeRate(
                    id=rate_dict['id'],
                    day=rate_dict['day'],
                    value=rate_dict['value'],
                    end_date=rate_dict['endDate']
                )
                exchange_rates.append(exchange_rate)
            
            # Create currency object
            currency = Currency(
                id=currency_dict['id'],
                name_fr=currency_dict['nameFr'],
                name_ar=currency_dict['nameAr'],
                unity=currency_dict['unity'],
                code=currency_dict['code'],
                exchange_rates=exchange_rates
            )
            currencies.append(currency)
        
        # Create final data structure
        return CurrencyData(
            success=json_data['success'],
            message=json_data['message'],
            data=currencies
        )

    @staticmethod
    def load_and_validate_from_file(file_path: str) -> CurrencyData:
        """
        Load JSON data from file and validate it
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            CurrencyData: Validated and parsed data structure
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is malformed
            ValueError: If data validation fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            
            return DataValidator.validate_complete_data(json_data)
        
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
