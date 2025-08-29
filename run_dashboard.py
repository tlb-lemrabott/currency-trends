#!/usr/bin/env python3
"""
Quick Dashboard Runner

This script runs the currency trends analysis dashboard immediately.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.visualization.dashboard import CurrencyDashboard


def main():
    """Run the dashboard"""
    print("=== Currency Trends Analysis Dashboard ===")
    print("Starting dashboard...")
    print("Dashboard will be available at: http://localhost:8050")
    print("Press Ctrl+C to stop the dashboard")
    print()
    
    try:
        # Create and run dashboard
        dashboard = CurrencyDashboard("Currency Trends Analysis Dashboard")
        dashboard.run(debug=True, port=8050)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install dash dash-bootstrap-components plotly")


if __name__ == "__main__":
    main()
