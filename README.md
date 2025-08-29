# Currency Trends Analysis

A comprehensive Python application for analyzing currency exchange rate trends and forecasting future movements using historical data from central banks.

## Features

- **Data Validation**: Robust JSON schema validation for currency exchange rate data
- **Trend Analysis**: Historical trend calculation and pattern identification
- **Forecasting**: Multiple forecasting models (ARIMA, Prophet, LSTM)
- **Visualization**: Interactive dashboards and charts
- **Reporting**: Automated PDF report generation

## Project Structure

```
currency-trends/
├── src/
│   ├── data/           # Data processing and validation
│   ├── analysis/       # Statistical analysis and forecasting
│   ├── visualization/  # Charts and dashboards
│   └── api/           # Web API endpoints
├── tests/             # Unit tests
├── data/              # Sample data files
├── docs/              # Documentation
└── requirements.txt   # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd currency-trends
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Validation

```python
from src.data.schema import DataValidator

# Load and validate data from file
currency_data = DataValidator.load_and_validate_from_file('data/sample_currency_data.json')

# Access validated data
for currency in currency_data.data:
    print(f"Currency: {currency.name_fr} ({currency.code})")
    for rate in currency.exchange_rates:
        print(f"  {rate.day}: {rate.value}")
```

### Database Operations

```python
from src.data.database import DatabaseManager

# Initialize database
db_manager = DatabaseManager("currency_trends.db")

# Insert currency data
db_manager.insert_currency_data(currency_data)

# Query exchange rates
usd_rates = db_manager.get_exchange_rates_by_currency("USD")
stats = db_manager.get_exchange_rate_statistics("USD")
```

### Data Ingestion

```python
from src.data.ingestion import DataIngestionPipeline

# Initialize pipeline
pipeline = DataIngestionPipeline(db_manager)

# Ingest from file
result = pipeline.ingest_from_file('data/sample_currency_data.json')

# Ingest from API
result = pipeline.ingest_from_api("https://api.example.com/currencies")

# Incremental update
result = pipeline.ingest_incremental_update(new_data)
```

### Data Preprocessing

```python
from src.data.preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Complete preprocessing pipeline
config = {
    'handle_missing': 'interpolate',
    'handle_outliers': 'iqr',
    'normalize': 'minmax',
    'add_indicators': True,
    'add_time_features': True,
    'lags': [1, 7, 30]
}

df, summary = preprocessor.preprocess_complete(currency_data, config)
```

### Running Tests

```bash
python -m pytest tests/ -v
```

### Running Examples

```bash
# Database example
python examples/database_example.py

# Ingestion example
python examples/ingestion_example.py

# Preprocessing example
python examples/preprocessing_example.py
```

## Development

This project follows:
- **Single Responsibility Principle**: Each module has a single, well-defined purpose
- **Separation of Concerns**: Clear separation between data, analysis, visualization, and API layers
- **Test-Driven Development**: Comprehensive unit tests for all components

## Issues and Progress

The project is being developed following a prioritized list of 20 issues:

- ✅ **Issue #1**: Project Environment Setup
- ✅ **Issue #2**: Data Schema Validation
- ✅ **Issue #3**: Database Schema Design
- ✅ **Issue #4**: Data Ingestion Pipeline
- ✅ **Issue #5**: Basic Data Preprocessing
- ⏳ **Issues #6-20**: Pending implementation

### Completed Features

1. **Data Validation**: Robust JSON schema validation with comprehensive error handling
2. **Database Management**: SQLite database with efficient indexing and CRUD operations
3. **Data Ingestion**: Support for file, directory, API, and incremental data loading
4. **Data Preprocessing**: Complete data cleaning pipeline with technical indicators
5. **Testing**: Comprehensive unit tests for all modules (60+ test cases)
6. **Documentation**: Detailed examples and usage instructions

## License

This project uses only free and open-source tools as requested.
