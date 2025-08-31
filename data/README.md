# Data Directory

This directory contains currency exchange rate data files for the Currency Trends Analysis application.

## ⚠️ Important Security Notice

**All data files are excluded from version control for security reasons.** The repository is public, so sensitive financial data should not be committed.

## Required Data Files

To run the application, you need to create the following files:

### 1. `sample_currency_data.json`
This is the main data file containing currency exchange rates. Use `data_template.json` as a reference for the expected structure.

### 2. `small_sample_currency_data.json` (Optional)
A smaller dataset for faster testing and development.

## Data Structure

The application expects data in the following Strapi-like format:

```json
{
  "success": true,
  "message": "Currency exchange rate data",
  "data": [
    {
      "id": 1,
      "attributes": {
        "name_fr": "Currency Name in French",
        "name_ar": "اسم العملة بالعربية",
        "unity": 100,
        "code": "XXX",
        "money_today_changes": {
          "data": [
            {
              "id": 1,
              "attributes": {
                "day": "2024-01-01",
                "value": "100.00",
                "end_date": "2024-01-02"
              }
            }
          ]
        }
      }
    }
  ]
}
```

## Setup Instructions

1. **Copy the template**: Use `data_template.json` as a starting point
2. **Add your data**: Replace the template data with your actual currency exchange rates
3. **Follow the structure**: Ensure your data matches the expected format
4. **Test the application**: Run the dashboard to verify data loading

## Supported Currencies

The application supports any currency codes (3-letter ISO codes) such as:
- USD (US Dollar)
- EUR (Euro)
- GBP (British Pound)
- JPY (Japanese Yen)
- CHF (Swiss Franc)

## Data Requirements

- **Date format**: YYYY-MM-DD
- **Value format**: String representation of decimal numbers
- **Time range**: Any historical period (recommended: at least 1 year)
- **Frequency**: Daily or monthly data points

## Security Best Practices

- Never commit real financial data to public repositories
- Use environment variables for sensitive configuration
- Keep data files in `.gitignore`
- Consider using encrypted data storage for production

## Troubleshooting

If you encounter data loading issues:
1. Check the JSON structure matches the template
2. Verify all required fields are present
3. Ensure date formats are correct
4. Check for duplicate entries in the data
