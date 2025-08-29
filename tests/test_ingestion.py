"""
Unit tests for data ingestion pipeline
"""

import pytest
import tempfile
import os
import json
import requests
from unittest.mock import Mock, patch
from src.data.ingestion import DataIngestionPipeline
from src.data.database import DatabaseManager
from src.data.schema import DataValidator


class TestDataIngestionPipeline:
    """Test cases for DataIngestionPipeline class"""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db_manager = DatabaseManager(db_path)
        yield db_manager
        
        # Cleanup
        os.unlink(db_path)

    @pytest.fixture
    def pipeline(self, temp_db):
        """Create ingestion pipeline with temporary database"""
        return DataIngestionPipeline(temp_db)

    @pytest.fixture
    def sample_data(self):
        """Sample currency data for testing"""
        return {
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

    def test_ingest_from_file_success(self, pipeline, sample_data):
        """Test successful file ingestion"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_file_path = f.name

        try:
            result = pipeline.ingest_from_file(temp_file_path)
            
            assert result['success'] is True
            assert result['source'] == 'file'
            assert result['currencies_processed'] == 1
            assert result['rates_processed'] == 1
            assert len(result['errors']) == 0
        finally:
            os.unlink(temp_file_path)

    def test_ingest_from_file_validate_only(self, pipeline, sample_data):
        """Test file ingestion with validate_only flag"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_file_path = f.name

        try:
            result = pipeline.ingest_from_file(temp_file_path, validate_only=True)
            
            assert result['success'] is True
            assert result['currencies_processed'] == 1
            assert result['rates_processed'] == 1
            
            # Check that data was not actually stored
            db_info = pipeline.get_ingestion_status()
            assert db_info['currency_count'] == 0
        finally:
            os.unlink(temp_file_path)

    def test_ingest_from_file_not_found(self, pipeline):
        """Test file ingestion with non-existent file"""
        result = pipeline.ingest_from_file("non_existent_file.json")
        
        assert result['success'] is False
        assert len(result['errors']) == 1
        assert "File not found" in result['errors'][0]

    def test_ingest_from_file_invalid_json(self, pipeline):
        """Test file ingestion with invalid JSON"""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file_path = f.name

        try:
            result = pipeline.ingest_from_file(temp_file_path)
            
            assert result['success'] is False
            assert len(result['errors']) == 1
            assert "Invalid JSON format" in result['errors'][0]
        finally:
            os.unlink(temp_file_path)

    def test_ingest_from_directory_success(self, pipeline, sample_data):
        """Test successful directory ingestion"""
        # Create temporary directory with JSON files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple JSON files
            for i in range(2):
                file_path = os.path.join(temp_dir, f"data_{i}.json")
                with open(file_path, 'w') as f:
                    json.dump(sample_data, f)
            
            result = pipeline.ingest_from_directory(temp_dir)
            
            assert result['success'] is True
            assert result['source'] == 'directory'
            assert result['files_processed'] == 2
            assert result['files_failed'] == 0
            assert result['currencies_processed'] == 2
            assert result['rates_processed'] == 2

    def test_ingest_from_directory_not_found(self, pipeline):
        """Test directory ingestion with non-existent directory"""
        result = pipeline.ingest_from_directory("non_existent_directory")
        
        assert result['success'] is False
        assert len(result['errors']) == 1
        assert "Directory not found" in result['errors'][0]

    def test_ingest_from_directory_mixed_files(self, pipeline, sample_data):
        """Test directory ingestion with mixed valid and invalid files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid JSON file
            valid_file = os.path.join(temp_dir, "valid.json")
            with open(valid_file, 'w') as f:
                json.dump(sample_data, f)
            
            # Create invalid JSON file
            invalid_file = os.path.join(temp_dir, "invalid.json")
            with open(invalid_file, 'w') as f:
                f.write("invalid json")
            
            result = pipeline.ingest_from_directory(temp_dir)
            
            assert result['success'] is False  # Should fail due to invalid file
            assert result['files_processed'] == 1
            assert result['files_failed'] == 1
            assert len(result['errors']) > 0

    @patch('requests.get')
    def test_ingest_from_api_success(self, mock_get, pipeline, sample_data):
        """Test successful API ingestion"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = sample_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = pipeline.ingest_from_api("https://api.example.com/currencies")
        
        assert result['success'] is True
        assert result['source'] == 'api'
        assert result['currencies_processed'] == 1
        assert result['rates_processed'] == 1
        assert len(result['errors']) == 0

    @patch('requests.get')
    def test_ingest_from_api_with_auth(self, mock_get, pipeline, sample_data):
        """Test API ingestion with authentication"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = sample_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = pipeline.ingest_from_api(
            "https://api.example.com/currencies",
            api_key="test_key"
        )
        
        assert result['success'] is True
        
        # Verify auth header was set
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[1]['headers']['Authorization'] == 'Bearer test_key'

    @patch('requests.get')
    def test_ingest_from_api_request_failed(self, mock_get, pipeline):
        """Test API ingestion with request failure"""
        # Mock failed request
        mock_get.side_effect = requests.RequestException("Connection failed")
        
        result = pipeline.ingest_from_api("https://api.example.com/currencies")
        
        assert result['success'] is False
        assert len(result['errors']) == 1
        assert "API request failed" in result['errors'][0]

    def test_ingest_incremental_update_success(self, pipeline, sample_data):
        """Test successful incremental update"""
        # First, insert some initial data
        initial_data = sample_data.copy()
        initial_data['data'][0]['exchangeRates'] = [
            {
                "id": 137058,
                "day": "2016-06-14",
                "value": "333.21",
                "endDate": "2016-06-15"
            }
        ]
        
        pipeline.ingest_from_file = Mock(return_value={'success': True})
        
        # Create new data with additional rates
        new_data = sample_data.copy()
        new_data['data'][0]['exchangeRates'].append({
            "id": 137059,
            "day": "2016-06-15",
            "value": "334.50",
            "endDate": "2016-06-16"
        })
        
        result = pipeline.ingest_incremental_update(new_data)
        
        assert result['success'] is True
        assert result['source'] == 'incremental'
        assert result['currencies_updated'] == 1
        assert result['rates_added'] == 2

    def test_validate_data_quality(self, pipeline, sample_data):
        """Test data quality validation"""
        from src.data.schema import DataValidator
        
        currency_data = DataValidator.validate_complete_data(sample_data)
        quality_report = pipeline.validate_data_quality(currency_data)
        
        assert quality_report['total_currencies'] == 1
        assert quality_report['total_rates'] == 1
        assert quality_report['date_range']['earliest'] == "2016-06-14"
        assert quality_report['date_range']['latest'] == "2016-06-14"
        assert quality_report['quality_score'] > 0

    def test_get_ingestion_status(self, pipeline):
        """Test getting ingestion status"""
        status = pipeline.get_ingestion_status()
        
        assert 'currency_count' in status
        assert 'exchange_rate_count' in status
        assert 'earliest_date' in status
        assert 'latest_date' in status
        assert 'database_path' in status

    def test_ingest_from_file_with_validation_error(self, pipeline):
        """Test file ingestion with validation error"""
        # Create data with validation error (missing required field)
        invalid_data = {
            "success": True,
            "message": "Success",
            "data": [
                {
                    "id": 1,
                    "nameFr": "Dollar US",
                    # Missing required fields
                    "exchangeRates": []
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_file_path = f.name

        try:
            result = pipeline.ingest_from_file(temp_file_path)
            
            assert result['success'] is False
            assert len(result['errors']) == 1
            assert "Data validation error" in result['errors'][0]
        finally:
            os.unlink(temp_file_path)
