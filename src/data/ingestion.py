"""
Data Ingestion Pipeline

This module handles automated data ingestion from various sources.
Implements single responsibility principle by focusing only on data ingestion.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import requests
from src.data.schema import DataValidator, CurrencyData
from src.data.database import DatabaseManager

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Pipeline for ingesting currency exchange rate data"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the ingestion pipeline
        
        Args:
            db_manager: Database manager instance for storing data
        """
        self.db_manager = db_manager
        self.validator = DataValidator()
    
    def ingest_from_file(self, file_path: str, validate_only: bool = False) -> Dict[str, Any]:
        """
        Ingest data from a JSON file
        
        Args:
            file_path: Path to the JSON file
            validate_only: If True, only validate without storing
            
        Returns:
            Dict: Result summary with success status and details
        """
        result = {
            'success': False,
            'source': 'file',
            'file_path': file_path,
            'currencies_processed': 0,
            'rates_processed': 0,
            'errors': []
        }
        
        try:
            # Validate and load data
            logger.info(f"Loading data from file: {file_path}")
            currency_data = self.validator.load_and_validate_from_file(file_path)
            
            result['currencies_processed'] = len(currency_data.data)
            result['rates_processed'] = sum(len(c.exchange_rates) for c in currency_data.data)
            
            if not validate_only:
                # Store in database
                success = self.db_manager.insert_currency_data(currency_data)
                if not success:
                    result['errors'].append("Failed to insert data into database")
                    return result
            
            result['success'] = True
            logger.info(f"Successfully processed {result['currencies_processed']} currencies with {result['rates_processed']} rates")
            
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        except ValueError as e:
            error_msg = f"Data validation error: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        
        return result
    
    def ingest_from_directory(self, directory_path: str, file_pattern: str = "*.json") -> Dict[str, Any]:
        """
        Ingest data from all matching files in a directory
        
        Args:
            directory_path: Path to the directory
            file_pattern: File pattern to match (default: *.json)
            
        Returns:
            Dict: Result summary with success status and details
        """
        result = {
            'success': True,
            'source': 'directory',
            'directory_path': directory_path,
            'files_processed': 0,
            'files_failed': 0,
            'currencies_processed': 0,
            'rates_processed': 0,
            'errors': []
        }
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                result['success'] = False
                result['errors'].append(f"Directory not found: {directory_path}")
                return result
            
            # Find all matching files
            files = list(directory.glob(file_pattern))
            logger.info(f"Found {len(files)} files matching pattern '{file_pattern}' in {directory_path}")
            
            for file_path in files:
                file_result = self.ingest_from_file(str(file_path))
                
                if file_result['success']:
                    result['files_processed'] += 1
                    result['currencies_processed'] += file_result['currencies_processed']
                    result['rates_processed'] += file_result['rates_processed']
                else:
                    result['files_failed'] += 1
                    result['errors'].extend([f"{file_path.name}: {error}" for error in file_result['errors']])
            
            if result['files_failed'] > 0:
                result['success'] = False
            
            logger.info(f"Directory ingestion completed: {result['files_processed']} files processed, {result['files_failed']} failed")
            
        except Exception as e:
            result['success'] = False
            result['errors'].append(f"Directory processing error: {e}")
            logger.error(f"Error processing directory {directory_path}: {e}")
        
        return result
    
    def ingest_from_api(self, api_url: str, api_key: Optional[str] = None, 
                       headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Ingest data from an API endpoint
        
        Args:
            api_url: URL of the API endpoint
            api_key: Optional API key for authentication
            headers: Optional additional headers
            
        Returns:
            Dict: Result summary with success status and details
        """
        result = {
            'success': False,
            'source': 'api',
            'api_url': api_url,
            'currencies_processed': 0,
            'rates_processed': 0,
            'errors': []
        }
        
        try:
            # Prepare request headers
            request_headers = headers or {}
            if api_key:
                request_headers['Authorization'] = f'Bearer {api_key}'
            
            # Make API request
            logger.info(f"Fetching data from API: {api_url}")
            response = requests.get(api_url, headers=request_headers, timeout=30)
            response.raise_for_status()
            
            # Parse JSON response
            json_data = response.json()
            
            # Validate and process data
            currency_data = self.validator.validate_complete_data(json_data)
            
            result['currencies_processed'] = len(currency_data.data)
            result['rates_processed'] = sum(len(c.exchange_rates) for c in currency_data.data)
            
            # Store in database
            success = self.db_manager.insert_currency_data(currency_data)
            if not success:
                result['errors'].append("Failed to insert data into database")
                return result
            
            result['success'] = True
            logger.info(f"Successfully ingested {result['currencies_processed']} currencies with {result['rates_processed']} rates from API")
            
        except requests.RequestException as e:
            error_msg = f"API request failed: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from API: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        except ValueError as e:
            error_msg = f"Data validation error: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        
        return result
    
    def ingest_incremental_update(self, new_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingest incremental updates to existing data
        
        Args:
            new_data: New data as file path or dictionary
            
        Returns:
            Dict: Result summary with success status and details
        """
        result = {
            'success': False,
            'source': 'incremental',
            'currencies_updated': 0,
            'rates_added': 0,
            'errors': []
        }
        
        try:
            # Load and validate new data
            if isinstance(new_data, str):
                currency_data = self.validator.load_and_validate_from_file(new_data)
            else:
                currency_data = self.validator.validate_complete_data(new_data)
            
            # Process each currency
            for currency in currency_data.data:
                # Check if currency exists
                existing_currency = self.db_manager.get_currency_by_code(currency.code)
                
                if existing_currency:
                    # Update existing currency
                    self.db_manager.insert_currency(currency)
                    result['currencies_updated'] += 1
                else:
                    # Insert new currency
                    self.db_manager.insert_currency(currency)
                    result['currencies_updated'] += 1
                
                # Process exchange rates
                for rate in currency.exchange_rates:
                    success = self.db_manager.insert_exchange_rate(currency.id, rate)
                    if success:
                        result['rates_added'] += 1
            
            result['success'] = True
            logger.info(f"Incremental update completed: {result['currencies_updated']} currencies updated, {result['rates_added']} rates added")
            
        except Exception as e:
            error_msg = f"Incremental update error: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        
        return result
    
    def validate_data_quality(self, currency_data: CurrencyData) -> Dict[str, Any]:
        """
        Validate data quality and provide quality metrics
        
        Args:
            currency_data: Currency data to validate
            
        Returns:
            Dict: Quality metrics and issues found
        """
        quality_report = {
            'total_currencies': len(currency_data.data),
            'total_rates': 0,
            'date_range': {'earliest': None, 'latest': None},
            'missing_dates': [],
            'duplicate_rates': [],
            'outliers': [],
            'quality_score': 0.0
        }
        
        all_dates = set()
        all_rates = []
        
        for currency in currency_data.data:
            quality_report['total_rates'] += len(currency.exchange_rates)
            
            for rate in currency.exchange_rates:
                all_dates.add(rate.day)
                all_rates.append(float(rate.value))
        
        # Calculate date range
        if all_dates:
            quality_report['date_range']['earliest'] = min(all_dates)
            quality_report['date_range']['latest'] = max(all_dates)
        
        # Check for outliers (simple statistical approach)
        if all_rates:
            import numpy as np
            rates_array = np.array(all_rates)
            mean_rate = np.mean(rates_array)
            std_rate = np.std(rates_array)
            
            # Identify outliers (beyond 3 standard deviations)
            outliers = rates_array[np.abs(rates_array - mean_rate) > 3 * std_rate]
            quality_report['outliers'] = outliers.tolist()
        
        # Calculate quality score (simplified)
        total_checks = 4  # currencies, rates, date_range, outliers
        passed_checks = 0
        
        if quality_report['total_currencies'] > 0:
            passed_checks += 1
        if quality_report['total_rates'] > 0:
            passed_checks += 1
        if quality_report['date_range']['earliest']:
            passed_checks += 1
        if len(quality_report['outliers']) == 0:
            passed_checks += 1
        
        quality_report['quality_score'] = passed_checks / total_checks
        
        return quality_report
    
    def get_ingestion_status(self) -> Dict[str, Any]:
        """
        Get current ingestion status and statistics
        
        Returns:
            Dict: Current database status and statistics
        """
        return self.db_manager.get_database_info()
