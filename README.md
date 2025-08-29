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

### Running Tests

```bash
python -m pytest tests/ -v
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
- 🔄 **Issue #3**: Database Schema Design (in progress)
- ⏳ **Issues #4-20**: Pending implementation

## License

This project uses only free and open-source tools as requested.
