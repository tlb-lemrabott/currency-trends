"""
Database Schema Design and Implementation

This module handles database operations for currency exchange rate data.
Implements single responsibility principle by focusing only on database operations.
"""

import sqlite3
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from src.data.schema import CurrencyData, Currency, ExchangeRate

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database manager for currency exchange rate data"""
    
    def __init__(self, db_path: str = "currency_trends.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._create_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create currencies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS currencies (
                    id INTEGER PRIMARY KEY,
                    name_fr TEXT NOT NULL,
                    name_ar TEXT NOT NULL,
                    unity INTEGER NOT NULL,
                    code TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create exchange_rates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exchange_rates (
                    id INTEGER PRIMARY KEY,
                    currency_id INTEGER NOT NULL,
                    day DATE NOT NULL,
                    value DECIMAL(10,4) NOT NULL,
                    end_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (currency_id) REFERENCES currencies (id),
                    UNIQUE(currency_id, day)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exchange_rates_currency_day 
                ON exchange_rates (currency_id, day)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exchange_rates_day 
                ON exchange_rates (day)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_currencies_code 
                ON currencies (code)
            """)
            
            conn.commit()
            logger.info("Database tables created successfully")
    
    def insert_currency(self, currency: Currency) -> bool:
        """
        Insert a currency into the database
        
        Args:
            currency: Currency object to insert
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO currencies 
                    (id, name_fr, name_ar, unity, code, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    currency.id,
                    currency.name_fr,
                    currency.name_ar,
                    currency.unity,
                    currency.code
                ))
                
                conn.commit()
                logger.info(f"Currency {currency.code} inserted successfully")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error inserting currency {currency.code}: {e}")
            return False
    
    def insert_exchange_rate(self, currency_id: int, exchange_rate: ExchangeRate) -> bool:
        """
        Insert an exchange rate into the database
        
        Args:
            currency_id: ID of the currency
            exchange_rate: ExchangeRate object to insert
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO exchange_rates 
                    (id, currency_id, day, value, end_date, created_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    exchange_rate.id,
                    currency_id,
                    exchange_rate.day,
                    float(exchange_rate.value),
                    exchange_rate.end_date
                ))
                
                conn.commit()
                logger.info(f"Exchange rate {exchange_rate.id} inserted successfully")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error inserting exchange rate {exchange_rate.id}: {e}")
            return False
    
    def insert_currency_data(self, currency_data: CurrencyData) -> bool:
        """
        Insert complete currency data into the database
        
        Args:
            currency_data: CurrencyData object containing all data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert currencies first
                for currency in currency_data.data:
                    cursor.execute("""
                        INSERT OR REPLACE INTO currencies 
                        (id, name_fr, name_ar, unity, code, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        currency.id,
                        currency.name_fr,
                        currency.name_ar,
                        currency.unity,
                        currency.code
                    ))
                
                # Insert exchange rates
                for currency in currency_data.data:
                    for exchange_rate in currency.exchange_rates:
                        cursor.execute("""
                            INSERT OR REPLACE INTO exchange_rates 
                            (id, currency_id, day, value, end_date, created_at)
                            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (
                            exchange_rate.id,
                            currency.id,
                            exchange_rate.day,
                            float(exchange_rate.value),
                            exchange_rate.end_date
                        ))
                
                conn.commit()
                logger.info("Currency data inserted successfully")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error inserting currency data: {e}")
            return False
    
    def get_currency_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Get currency by code
        
        Args:
            code: Currency code (e.g., 'USD', 'EUR')
            
        Returns:
            Optional[Dict]: Currency data or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM currencies WHERE code = ?
                """, (code,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Error getting currency {code}: {e}")
            return None
    
    def get_exchange_rates_by_currency(self, currency_code: str, 
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get exchange rates for a specific currency
        
        Args:
            currency_code: Currency code
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            List[Dict]: List of exchange rate data
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT er.*, c.code as currency_code, c.name_fr as currency_name
                    FROM exchange_rates er
                    JOIN currencies c ON er.currency_id = c.id
                    WHERE c.code = ?
                """
                params = [currency_code]
                
                if start_date:
                    query += " AND er.day >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND er.day <= ?"
                    params.append(end_date)
                
                query += " ORDER BY er.day"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Error getting exchange rates for {currency_code}: {e}")
            return []
    
    def get_all_currencies(self) -> List[Dict[str, Any]]:
        """
        Get all currencies from database
        
        Returns:
            List[Dict]: List of all currencies
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM currencies ORDER BY code
                """)
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Error getting all currencies: {e}")
            return []
    
    def get_exchange_rate_statistics(self, currency_code: str) -> Dict[str, Any]:
        """
        Get statistical summary for a currency
        
        Args:
            currency_code: Currency code
            
        Returns:
            Dict: Statistical summary including min, max, avg, count
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as count,
                        MIN(value) as min_value,
                        MAX(value) as max_value,
                        AVG(value) as avg_value,
                        MIN(day) as first_date,
                        MAX(day) as last_date
                    FROM exchange_rates er
                    JOIN currencies c ON er.currency_id = c.id
                    WHERE c.code = ?
                """, (currency_code,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return {}
                
        except sqlite3.Error as e:
            logger.error(f"Error getting statistics for {currency_code}: {e}")
            return {}
    
    def delete_currency_data(self, currency_code: str) -> bool:
        """
        Delete all data for a specific currency
        
        Args:
            currency_code: Currency code to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get currency ID first
                cursor.execute("SELECT id FROM currencies WHERE code = ?", (currency_code,))
                row = cursor.fetchone()
                
                if not row:
                    logger.warning(f"Currency {currency_code} not found")
                    return False
                
                currency_id = row['id']
                
                # Delete exchange rates first (due to foreign key constraint)
                cursor.execute("DELETE FROM exchange_rates WHERE currency_id = ?", (currency_id,))
                
                # Delete currency
                cursor.execute("DELETE FROM currencies WHERE id = ?", (currency_id,))
                
                conn.commit()
                logger.info(f"Currency {currency_code} and all its data deleted successfully")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error deleting currency {currency_code}: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database statistics and information
        
        Returns:
            Dict: Database information including table sizes, etc.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get currency count
                cursor.execute("SELECT COUNT(*) as currency_count FROM currencies")
                currency_count = cursor.fetchone()['currency_count']
                
                # Get exchange rate count
                cursor.execute("SELECT COUNT(*) as rate_count FROM exchange_rates")
                rate_count = cursor.fetchone()['rate_count']
                
                # Get date range
                cursor.execute("""
                    SELECT MIN(day) as earliest_date, MAX(day) as latest_date 
                    FROM exchange_rates
                """)
                date_range = cursor.fetchone()
                
                return {
                    'currency_count': currency_count,
                    'exchange_rate_count': rate_count,
                    'earliest_date': date_range['earliest_date'],
                    'latest_date': date_range['latest_date'],
                    'database_path': self.db_path
                }
                
        except sqlite3.Error as e:
            logger.error(f"Error getting database info: {e}")
            return {}
